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
	_, err = vs.LoadPartial(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	return net
}

func runValidate() {
	vs := nn.NewVarStore(Device)
	net := loadModel("./model/hubmap-epoch4.gt", vs)
	doValidate(net, Device)
}

func runCheckModel() {
	vs := nn.NewVarStore(Device)
	net := loadModel("./checkpoint/hubmap.gt", vs)
	image := "./input/tile/image/0486052bb_190.png"

	imgTs, err := vision.Load(image)
	if err != nil {
		log.Fatal(err)
	}

	x := imgTs.MustDiv1(ts.FloatScalar(255.0), true)
	input := x.MustUnsqueeze(0, true)
	out := net.ForwardT(input, false)

	fmt.Printf("%v", out)

	mask := out.MustSqueeze1(0, true)

	fmt.Printf("mask shape: %v\n", mask.MustSize())

	err = vision.Save(mask, "./test.png")
	if err != nil {
		log.Fatal(err)
	}
}
