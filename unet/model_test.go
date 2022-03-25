package unet_test

import (
	"fmt"
	"testing"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"

	"github.com/sugarme/iseg/unet"
)

func TestNewUNet(t *testing.T) {
	device := gotch.CPU
	vs := nn.NewVarStore(device)
	net := unet.DefaultUNet(vs.Root())

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
	fmt.Printf("image: %i\n", image)

	for i := 0; i < 100; i++ {
		logit := net.ForwardT(image, false).MustTotype(gotch.Double, true)
		logit.MustDrop()
		t.Logf("%02d - processed\n", i)
	}
	// loss := criterionBinaryCrossEntropy(logit, mask)
	// fmt.Printf("mask: %v\n", mask.MustSize())
	// fmt.Printf("image: %v\n", image.MustSize())
	// fmt.Printf("logit: %v\n", logit.MustSize())
	// l := loss.Float64Values()[0]
	// fmt.Printf("%i", logit)

	t.Error("stop")
}
