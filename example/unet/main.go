package main

import (
	"fmt"
	"log"
	"sort"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"

	"github.com/sugarme/iseg/unet"
)

func loadResNet34() {
	// Create the model and load the weights from the file.

	in := vision.NewImageNet()
	vs := nn.NewVarStore(gotch.CPU)
	var net ts.ModuleT
	net = vision.ResNet34(vs.Root(), in.ClassCount())
	err := vs.Load("./resnet34.ot")
	if err != nil {
		log.Fatal(err)
	}

	printVars(vs)

	fmt.Println(net)
}

func checkModel() {
	device := gotch.CPU
	vs := nn.NewVarStore(device)

	net := unet.DefaultUNet(vs.Root())

	// printVars(vs)

	_, err := vs.LoadPartial("./resnet34.ot")
	if err != nil {
		log.Fatal(err)
	}

	// Pytorch equivalent to `np.random.choice()`
	// Ref. https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/13
	batchSize := int64(36)
	imageSize := int64(256)
	// a := []int64{0, 1}                       // values
	// p := []float64{0.5, 0.5}                 // probability
	// n := (batchSize * imageSize * imageSize) // size
	// replace := true
	// aTs := ts.MustOfSlice(a)
	// pTs := ts.MustOfSlice(p)
	// idx := pTs.MustMultinomial(n, replace, true)
	// mask := aTs.MustIndex([]ts.Tensor{*idx}, true).MustView([]int64{batchSize, imageSize, imageSize}, true).MustTotype(gotch.Double, true)

	image := ts.MustRand([]int64{batchSize, 3, imageSize, imageSize}, gotch.Float, gotch.CPU)
	for i := 0; i < 100; i++ {
		ts.NoGrad(func() {
			logit := net.ForwardT(image, false)
			// loss := criterionBinaryCrossEntropy(logit, mask)
			// fmt.Printf("mask: %v\n", mask.MustSize())
			// fmt.Printf("image: %v\n", image.MustSize())
			// fmt.Printf("logit: %v\n", logit.MustSize())
			// l := loss.Float64Values()[0]
			// fmt.Printf("%02d - Loss: %v\n", i, l)

			logit.MustDrop()
			fmt.Printf("Done %02d\n", i)
			// loss.MustDrop()
		})
	}
}

// printVars print variables sorted by name
func printVars(vs *nn.VarStore) {
	vars := vs.Variables()
	names := make([]string, 0, len(vars))
	for n := range vars {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		fmt.Printf("%v \t\t %v\n", n, vars[n].MustSize())
	}
}

func main() {
	checkModel()
	// loadResNet34()
}
