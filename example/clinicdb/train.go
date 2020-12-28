package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"path/filepath"

	// "strings"
	"time"

	"github.com/sugarme/iseg/example/hubmap/dutil"
	"github.com/sugarme/iseg/metric"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	// "github.com/sugarme/gotch/vision"

	"github.com/sugarme/iseg/unet"
)

func loadWeights(vs *nn.VarStore, fpath string, from string) {
	modelPath, err := filepath.Abs(fpath)
	if err != nil {
		log.Fatal(err)
	}

	switch from {
	case "checkpoint":
		err = vs.Load(modelPath)
		if err != nil {
			log.Fatal(err)
		}
	case "scratch":
		_, err = vs.LoadPartial(modelPath)
		if err != nil {
			log.Fatal(err)
		}
	default:
		err := fmt.Errorf("Invalid load option. Expected 'checkpoint' or 'scratch'. Got: %v\n", from)
		panic(err)
	}
}

func runTrain() {
	var (
		err error
		// bestDice float64 = 0
	)

	vs := nn.NewVarStore(Device)
	net := unet.NewUNetOriginal(vs.Root())
	loadWeights(vs, ModelPath, ModelFrom)

	var opt *nn.Optimizer
	switch OptStr {
	case "SGD":
		opt, err = nn.DefaultSGDConfig().Build(vs, LR)
		if err != nil {
			log.Fatal(err)
		}
	case "Adam":
		opt, err = nn.DefaultAdamConfig().Build(vs, LR)
		if err != nil {
			log.Fatal(err)
		}
	default:
		err = fmt.Errorf("Unspecified/Invalid Optimizer option: '%v'.\n", OptStr)
		log.Fatal(err)
	}

	imgPath := fmt.Sprintf("%v/image", DataPath)
	files, err := ioutil.ReadDir(imgPath)
	if err != nil {
		log.Fatal(err)
	}

	var trainFiles []string
	for i, f := range files {
		if i < 25 {
			continue
		}
		trainFiles = append(trainFiles, f.Name())
	}

	trainDS := NewDataset(trainFiles)
	s, err := dutil.NewBatchSampler(trainDS.Len(), BatchSize, true, true)
	if err != nil {
		log.Fatal(err)
	}
	trainDL, err := dutil.NewDataLoader(trainDS, s)
	if err != nil {
		log.Fatal(err)
	}

	// Epochs
	for e := 0; e < Epochs; e++ {
		start := time.Now()
		count := 0
		trainDL.Reset()
		var losses []float64 = []float64{}

		for trainDL.HasNext() {
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

			input := imgTs.MustTo(Device, true)
			logit := net.ForwardT(input, true)
			pred := logit.MustTotype(gotch.Double, true)
			target := maskTs.MustTo(Device, true)

			loss := metric.BCEWithLogitsLoss(pred, target)
			// loss := LossFunc(pred, target)
			pred.MustDrop()
			target.MustDrop()

			opt.BackwardStep(loss)
			// opt.ZeroGrad()
			lossVal := loss.Float64Values()[0]
			losses = append(losses, lossVal)
			loss.MustDrop()
		}

		var tloss float64
		var lossSum float64
		for _, loss := range losses {
			lossSum += loss
		}
		tloss = lossSum / float64(len(losses))

		// validate
		vloss, dice := doValidate(net, Device)
		fmt.Printf("Epoch %02d\t train loss: %6.4f\t valid loss: %6.4f\t dice: %6.4f\t Taken time: %0.2fMin\n", e, tloss, vloss, dice, time.Since(start).Minutes())

		// save model checkpoint
		// if dice > bestDice {
		// bestDice = dice
		// weightFile := fmt.Sprintf("./checkpoint/clinicdb-%v.gt", time.Now().Unix())
		// err = vs.Save(weightFile)
		// if err != nil {
		// log.Fatal(err)
		// }
		// }
	}

	// save model checkpoint
	weightFile := fmt.Sprintf("./checkpoint/clinicdb-%v.gt", time.Now().Unix())
	err = vs.Save(weightFile)
	if err != nil {
		log.Fatal(err)
	}

}

func doValidate(net ts.ModuleT, device gotch.Device) (loss, dice float64) {
	var testFiles []string

	imgPath := fmt.Sprintf("%v/image", DataPath)
	files, err := ioutil.ReadDir(imgPath)
	if err != nil {
		log.Fatal(err)
	}

	for i, f := range files {
		if i > 24 {
			break
		}
		testFiles = append(testFiles, f.Name())
	}

	testDS := NewDataset(testFiles)
	s, err := dutil.NewBatchSampler(testDS.Len(), 25, true, false) // no shuffle
	if err != nil {
		log.Fatal(err)
	}
	testDL, err := dutil.NewDataLoader(testDS, s)
	if err != nil {
		log.Fatal(err)
	}

	var (
		losses []float64
		dices  []float64
	)
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
		imgTs := ts.MustStack(img, 0).MustTo(device, true)
		for _, x := range img {
			x.MustDrop()
		}
		maskTs := ts.MustStack(mask, 0).MustTo(device, true)
		for _, x := range mask {
			x.MustDrop()
		}

		var logit *ts.Tensor
		ts.NoGrad(func() {
			logit = net.ForwardT(imgTs, false).MustTotype(gotch.Double, true)
		})
		prob := logit.MustSigmoid(false).MustTotype(gotch.Float, true)

		// Loss
		// loss := criterionBinaryCrossEntropy(logit, mask)
		loss := metric.BCEWithLogitsLoss(logit, maskTs)
		// loss := LossFunc(logit, mask)
		lossVal := loss.Float64Values()[0]
		losses = append(losses, lossVal)

		// Dice score
		// threshold := 0.5
		// dice := DiceScore(prob, mask, threshold)
		// dice := DiceLoss(prob.MustSqueeze1(1, false), mask)
		// dice := metric.IoU(logit, maskTs)
		dice := metric.DiceCoeffBatch(prob, maskTs)
		fmt.Printf("dice: %v\n", dice)
		dices = append(dices, dice)

		maskTs.MustDrop()
		imgTs.MustDrop()
		logit.MustDrop()
		loss.MustDrop()
		prob.MustDrop()
	}

	loss = avg(losses)
	dice = avg(dices)

	return loss, dice
}

func avg(input []float64) float64 {
	var sum float64
	for _, v := range input {
		sum += v
	}

	return sum / float64(len(input))
}
