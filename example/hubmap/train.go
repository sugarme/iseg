package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"path/filepath"

	"github.com/sugarme/iseg/example/hubmap/dutil"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

func loadResNetUnetModel(vs *nn.VarStore) nn.FuncT {
	modelPath, err := filepath.Abs(ModelPath)
	if err != nil {
		log.Fatal(err)
	}

	// device := gotch.CPU
	// vs := nn.NewVarStore(device)
	net := ResNet34Unet(vs.Root(), true)

	_, err = vs.LoadPartial(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	return net
}

func fakeInput() (input, mask *ts.Tensor) {
	name := "0486052bb"
	var images, masks []ts.Tensor
	tileImgPath := fmt.Sprintf("%v/tile/image", DataPath)
	tileMaskPath := fmt.Sprintf("%v/tile/mask", DataPath)
	for i := 0; i < 24; i++ {
		imgPath := fmt.Sprintf("%v/%v_%03d.png", tileImgPath, name, i)
		maskPath := fmt.Sprintf("%v/%v_%03d.png", tileMaskPath, name, i)
		ix, err := vision.Load(imgPath)
		if err != nil {
			panic(err)
		}
		mx, err := vision.Load(maskPath)
		if err != nil {
			panic(err)
		}
		images = append(images, *ix)
		masks = append(masks, *mx)
	}

	imgTs := ts.MustStack(images, 0)
	for _, x := range images {
		x.MustDrop()
	}

	maskTs := ts.MustStack(masks, 0)
	for _, x := range masks {
		x.MustDrop()
	}

	return imgTs, maskTs
}

func runTrain() {
	tileImgPath := fmt.Sprintf("%v/tile/image", DataPath)
	files, err := ioutil.ReadDir(tileImgPath)
	if err != nil {
		log.Fatal(err)
	}

	var trainFiles []string
	for _, f := range files {
		trainFiles = append(trainFiles, f.Name())
	}

	vs := nn.NewVarStore(Device)
	net := loadResNetUnetModel(vs)

	trainDS := NewHubmapDataset(trainFiles)
	s, err := dutil.NewBatchSampler(trainDS.Len(), BatchSize, true, true)
	if err != nil {
		log.Fatal(err)
	}
	trainDL, err := dutil.NewDataLoader(trainDS, s)
	if err != nil {
		log.Fatal(err)
	}

	var si *SI
	si = CPUInfo()
	fmt.Printf("Total RAM (MB):\t %8.2f\n", float64(si.TotalRam)/1024)
	fmt.Printf("Used RAM (MB):\t %8.2f\n", float64(si.TotalRam-si.FreeRam)/1024)
	startRAM := si.TotalRam - si.FreeRam

	count := 0
	for trainDL.HasNext() {
		// Validate
		if count != 0 && count%ValidateSize == 0 {
			doValidate(net, Device)
		}
		s, err := trainDL.Next()
		if err != nil {
			log.Fatal(err)
		}

		count++

		var img, mask []ts.Tensor
		for _, i := range s.([]ImageMask) {
			img = append(img, i.image)
			mask = append(mask, i.mask)
		}
		imgTs := ts.MustStack(img, 0)
		for _, x := range img {
			x.MustDrop()
		}
		maskTs := ts.MustStack(mask, 0)
		for _, x := range mask {
			x.MustDrop()
		}

		lr := 0.001
		var opt *nn.Optimizer
		switch OptStr {
		case "SGD":
			opt, err = nn.DefaultSGDConfig().Build(vs, lr)
			if err != nil {
				log.Fatal(err)
			}
		case "Adam":
			opt, err = nn.DefaultAdamConfig().Build(vs, lr)
			if err != nil {
				log.Fatal(err)
			}
		default:
			err = fmt.Errorf("Unspecified/Invalid Optimizer option: '%v'.\n", OptStr)
			log.Fatal(err)
		}

		if Device == gotch.CPU {
			si = CPUInfo()
			startRAM = si.TotalRam - si.FreeRam
		}
		input := imgTs.MustTo(Device, true)
		logit := net.ForwardT(input, true)
		input.MustDrop()
		pred := logit.MustTotype(gotch.Double, true)
		target := maskTs.MustTo(Device, true)

		loss := criterionBinaryCrossEntropy(pred, target)
		pred.MustDrop()
		target.MustDrop()

		opt.BackwardStep(loss)
		// opt.ZeroGrad()
		fmt.Printf("loss: %v", loss)
		loss.MustDrop()

		if Device == gotch.CPU {
			si = CPUInfo()
			fmt.Printf("Batch %v\t Used: [%8.2f MiB]\n", count, (float64(si.TotalRam-si.FreeRam)-float64(startRAM))/1024)
		}
	}
}

func doValidate(net nn.FuncT, device gotch.Device) {
	mockName := "0486052bb"
	var testFiles []string
	for i := 0; i < 347; i++ {
		n := fmt.Sprintf("%v_%03d.png", mockName, i+1)
		testFiles = append(testFiles, n)
	}

	testDS := NewHubmapDataset(testFiles)
	s, err := dutil.NewBatchSampler(testDS.Len(), BatchSize, true, false) // no shuffle
	if err != nil {
		log.Fatal(err)
	}
	testDL, err := dutil.NewDataLoader(testDS, s)
	if err != nil {
		log.Fatal(err)
	}
	for testDL.HasNext() {
		s, err := testDL.Next()
		if err != nil {
			log.Fatal(err)
		}

		var img, mask []ts.Tensor
		for _, i := range s.([]ImageMask) {
			img = append(img, i.image)
			mask = append(mask, i.mask)
		}
		imgTs := ts.MustStack(img, 0)
		for _, x := range img {
			x.MustDrop()
		}
		maskTs := ts.MustStack(mask, 0)
		for _, x := range mask {
			x.MustDrop()
		}

		ts.NoGrad(func() {
			logit := net.ForwardT(imgTs.MustTo(device, true), true).MustTotype(gotch.Double, true)
			// dice := DiceScore(logit, maskTs.MustTo(device, true))
			loss := criterionBinaryCrossEntropy(logit, maskTs.MustTo(device, true))
			fmt.Printf("Test loss: %.5f", loss)
			// fmt.Printf("Dice Score: %v", dice)
			imgTs.MustDrop()
			maskTs.MustDrop()
			logit.MustDrop()
			loss.MustDrop()
		})
	}
}
