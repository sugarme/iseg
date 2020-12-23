package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
	"github.com/sugarme/iseg/unet"
)

func loadModel(file string, vs *nn.VarStore) ts.ModuleT {
	modelPath, err := filepath.Abs(file)
	if err != nil {
		log.Fatal(err)
	}

	net := unet.DefaultUNet(vs.Root())
	if file == "./model/resnet34.ot" {
		_, err = vs.LoadPartial(modelPath)
		if err != nil {
			log.Fatal(err)
		}
		return net
	}

	err = vs.Load(modelPath)
	if err != nil {
		log.Fatal(err)
	}
	return net

}

func runValidate() {
	vs := nn.NewVarStore(Device)
	net := loadModel(ModelPath, vs)
	loss, dice, tp, tn := doValidate(net, Device)
	fmt.Printf("Loss: %6.4f\t Dice: %6.4f\t TP: %6.4f\t TN: %6.4f\n", loss, dice, tp, tn)
}

func runCheckModel() {
	vs := nn.NewVarStore(Device)
	// net := loadModel("./checkpoint/hubmap.gt", vs)
	net := loadModel(ModelPath, vs)
	image := "./input/tile/image/2f6ecfcdf_014.png"
	mask := "./input/tile/mask/2f6ecfcdf_014.png"

	imgTs, err := vision.Load(image)
	if err != nil {
		log.Fatal(err)
	}

	maskTs, err := vision.Load(mask)
	if err != nil {
		log.Fatal(err)
	}

	maskGray, err := rgb2GrayScale(maskTs)
	if err != nil {
		log.Fatal(err)
	}
	// targetPixels := maskGray.MustDiv1(ts.FloatScalar(255.0), false)
	// fmt.Printf("%0.2f", targetPixels)
	mOut := maskGray.MustUnsqueeze(0, true)
	err = vision.Save(mOut, "./test-mask.png")
	if err != nil {
		panic(err)
	}
	maskTs.MustDrop()

	x := imgTs.MustDiv1(ts.FloatScalar(255.0), true)
	input := x.MustUnsqueeze(0, true)
	logit := net.ForwardT(input, false)
	prob := logit.MustSigmoid(true)
	// fmt.Printf("%0.2f", prob)

	threshold := 0.1
	// out1 := out.MustExp(true).MustGt(ts.FloatScalar(0.2), true).MustSqueeze1(0, true)
	pred := prob.MustGt(ts.FloatScalar(threshold), true)
	predPixels := pred.MustMul1(ts.FloatScalar(255.0), true)
	// fmt.Printf("%v", predPixels)

	// fmt.Printf("mask shape: %v\n", mask.MustSize())
	err = vision.Save(predPixels, "./test.png")
	if err != nil {
		log.Fatal(err)
	}
}
