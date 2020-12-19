package main

import (
	"log"
	"path/filepath"

	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/iseg/unet"
)

func runValidate() {
	modelPath, err := filepath.Abs("./model/hubmap-epoch4.gt")
	if err != nil {
		log.Fatal(err)
	}

	vs := nn.NewVarStore(Device)
	net := unet.DefaultUNet(vs.Root())
	_, err = vs.LoadPartial(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	doValidate(net, Device)
}
