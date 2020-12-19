package main

import (
	"fmt"
	"log"
	"path/filepath"
	"reflect"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

func loadModel() {
	// Create the model and load the weights from the file.
	modelPath, err := filepath.Abs(ModelPath)
	if err != nil {
		log.Fatal(err)
	}

	in := vision.NewImageNet()
	vs := nn.NewVarStore(gotch.CPU)
	var net ts.ModuleT
	net = vision.ResNet34(vs.Root(), in.ClassCount())
	err = vs.Load(modelPath)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("ResNet34 weights loaded.")
	for n := range vs.Vars.NamedVariables {
		fmt.Println(n)
	}

	fmt.Println(net)
}

func downSample(path *nn.Path, cIn, cOut, stride int64) ts.ModuleT {
	if stride != 1 || cIn != cOut {
		seq := nn.SeqT()
		seq.Add(conv2d(path.Sub("0"), cIn, cOut, 1, 0, stride))
		seq.Add(nn.BatchNorm2D(path.Sub("1"), cOut, nn.DefaultBatchNormConfig()))

		return seq
	}
	return nn.SeqT()
}

func basicBlock(path *nn.Path, cIn, cOut, stride int64) ts.ModuleT {
	conv1 := conv2d(path.Sub("conv1"), cIn, cOut, 3, 1, stride)
	bn1 := nn.BatchNorm2D(path.Sub("bn1"), cOut, nn.DefaultBatchNormConfig())
	conv2 := conv2d(path.Sub("conv2"), cOut, cOut, 3, 1, 1)
	bn2 := nn.BatchNorm2D(path.Sub("bn2"), cOut, nn.DefaultBatchNormConfig())
	downsample := downSample(path.Sub("downsample"), cIn, cOut, stride)

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		c1 := xs.Apply(conv1)
		bn1Ts := c1.ApplyT(bn1, train)
		c1.MustDrop()
		relu := bn1Ts.MustRelu(true)
		c2 := relu.Apply(conv2)
		relu.MustDrop()
		bn2Ts := c2.ApplyT(bn2, train)
		c2.MustDrop()

		dsl := xs.ApplyT(downsample, train)
		dslAdd := dsl.MustAdd(bn2Ts, true)
		bn2Ts.MustDrop()
		res := dslAdd.MustRelu(true)

		return res
	})
}

func layerZero(p *nn.Path) ts.ModuleT {
	conv1 := conv2d(p.Sub("conv1"), 3, 64, 7, 3, 2)
	bn1 := nn.BatchNorm2D(p.Sub("bn1"), 64, nn.DefaultBatchNormConfig())
	layer0 := nn.SeqT()
	layer0.Add(conv1)
	layer0.Add(bn1)
	layer0.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))
	layer0.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustMaxPool2d([]int64{3, 3}, []int64{2, 2}, []int64{1, 1}, []int64{1, 1}, false, false)
	}))

	return layer0
}

func basicLayer(path *nn.Path, cIn, cOut, stride, cnt int64) ts.ModuleT {
	layer := nn.SeqT()
	layer.Add(basicBlock(path.Sub("0"), cIn, cOut, stride))
	for blockIndex := 1; blockIndex < int(cnt); blockIndex++ {
		layer.Add(basicBlock(path.Sub(fmt.Sprint(blockIndex)), cOut, cOut, 1))
	}

	return layer
}

func conv2d(p *nn.Path, cIn, cOut, ksize, padding, stride int64) *nn.Conv2D {
	config := nn.DefaultConv2DConfig()
	config.Stride = []int64{stride, stride}
	config.Padding = []int64{padding, padding}

	return nn.NewConv2D(p, cIn, cOut, ksize, config)
}

func conv2dNoBias(p *nn.Path, cIn, cOut, ksize, padding, stride int64) *nn.Conv2D {
	config := nn.DefaultConv2DConfig()
	config.Bias = false
	config.Stride = []int64{stride, stride}
	config.Padding = []int64{padding, padding}

	return nn.NewConv2D(p, cIn, cOut, ksize, config)
}

func squeezeExcite(p *nn.Path, cIn, reduction int64) ts.ModuleT {
	// Channel squeeze excite
	chanSeq := nn.SeqT()
	chanSeq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustAdaptiveAvgPool2d([]int64{1, 1}, false)
	}))
	chanSeq.Add(conv2d(p, cIn, cIn/reduction, 1, 0, 1))
	chanSeq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))
	chanSeq.Add(conv2d(p, cIn/reduction, cIn, 1, 0, 1))
	chanSeq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustSigmoid(false)
	}))

	// Spatial squeeze excite
	spatSeq := nn.SeqT()
	spatSeq.Add(conv2d(p, cIn, 1, 1, 0, 1))
	chanSeq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustSigmoid(false)
	}))

	return nn.NewFuncT(func(x *ts.Tensor, train bool) *ts.Tensor {
		fwdChan := x.ApplyT(chanSeq, train)
		xChan := x.MustMul(fwdChan, false)
		fwdChan.MustDrop()
		fwdSpat := x.ApplyT(spatSeq, train)
		xSpat := x.MustMul(fwdSpat, false)
		fwdSpat.MustDrop()

		sum := xChan.MustAdd(xSpat, true)
		xSpat.MustDrop()

		return sum
	})
}

// decodeBlock creates an unet decode block.
func decodeLayer(p *nn.Path, cIn, cOut int64, skip *ts.Tensor) ts.ModuleT {
	bnConfig := nn.DefaultBatchNormConfig()
	bnConfig.Eps = 0.001

	conv1 := nn.SeqT()
	conv1.Add(conv2dNoBias(p, cIn, cOut, 3, 1, 1))
	conv1.Add(nn.BatchNorm2D(p.Sub("conv1_bn"), cOut, bnConfig))
	conv1.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	attn1 := squeezeExcite(p, cIn, 16)

	conv2 := nn.SeqT()
	conv2.Add(conv2dNoBias(p, cOut, cOut, 3, 1, 1))
	conv2.Add(nn.BatchNorm2D(p.Sub("conv2_bn"), cOut, bnConfig))
	conv2.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))
	attn2 := squeezeExcite(p, cOut, 16)

	return nn.NewFuncT(func(x *ts.Tensor, train bool) *ts.Tensor {
		// concat
		var catTs *ts.Tensor = x
		if skip != nil {
			catTs = ts.MustCat([]ts.Tensor{*x, *skip}, 1)
		}

		fwdAttn1 := catTs.ApplyT(attn1, train)
		fwdConv1 := fwdAttn1.ApplyT(conv1, train)
		fwdConv2 := fwdConv1.ApplyT(conv2, train)
		out := fwdConv2.ApplyT(attn2, train)

		if skip != nil {
			catTs.MustDrop()
		}
		fwdAttn1.MustDrop()
		fwdConv1.MustDrop()
		fwdConv2.MustDrop()
		return out
	})
}

// centerBlock creates center block for Unet model.
func centerBlock(p *nn.Path, train bool) *nn.SequentialT {
	bnConfig := nn.DefaultBatchNormConfig()
	bnConfig.Eps = 0.001

	seq := nn.SeqT()
	seq.Add(conv2dNoBias(p, 512, 512, 11, 5, 1))
	seq.Add(nn.BatchNorm2D(p, 512, bnConfig))
	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	return seq
}

// interpolation using `nearest` algorithm
func upsample(x, ref *ts.Tensor) *ts.Tensor {
	xSize := x.MustSize()
	refSize := ref.MustSize()
	if reflect.DeepEqual(xSize[2:], refSize[2:]) {
		return x
	}

	return x.MustUpsampleNearest2d(refSize[2:], nil, nil, false)
}

// ResNet34Unet creates ResNet34-Unet model.
func ResNet34Unet(p *nn.Path, train bool) nn.FuncT {
	// layer0										decoder4
	//		layer1							decoder3
	//			layer2					decoder2
	//				layer3			decoder1
	//					layer4	decoder0
	//							center
	// encoder
	layer0 := layerZero(p) // NOTE. `conv1` and `bn1` are at root of pretrained model
	layer1 := basicLayer(p.Sub("layer1"), 64, 64, 1, 3)
	layer2 := basicLayer(p.Sub("layer2"), 64, 128, 2, 4)
	layer3 := basicLayer(p.Sub("layer3"), 128, 256, 2, 6)
	layer4 := basicLayer(p.Sub("layer4"), 256, 512, 2, 3)

	// x should have shape = [batch_size, C, H, W]
	return nn.NewFuncT(func(x *ts.Tensor, t bool) *ts.Tensor {
		// E.g. x [4 3 256 256]
		x0 := x.ApplyT(layer0, train)  // 	[4 	64 64 64]
		x1 := x0.ApplyT(layer1, train) //		[4 	64 64 64]
		x2 := x1.ApplyT(layer2, train) //		[4 128 32 32]
		x3 := x2.ApplyT(layer3, train) //		[4 256 16 16]
		x4 := x3.ApplyT(layer4, train) //		[4 512 	8	 8]

		// center
		center := centerBlock(p.Sub("center"), train)
		z := x4.ApplyT(center, train) //		[4 512  8  8]

		// NOTE: for batchSize = 4; imageSize= 256 x 256
		// z0: [4 512 8 8]
		// z1: [4 256 16 16]
		// z2: [4 128 32 32]
		// z3: [4 64 64 64]
		// z4: [4 32 64 64]

		// decoder
		ref0 := upsample(z, x3)                                      // upsample to 16x16
		decode0 := decodeLayer(p.Sub("decode0"), 256+512, 256, ref0) //
		z0 := x3.ApplyT(decode0, train)                              // [4 256 16 16]
		ref1 := upsample(z0, x2)                                     // upsample to 32x32
		decode1 := decodeLayer(p.Sub("decode1"), 128+256, 128, ref1) //
		z1 := x2.ApplyT(decode1, train)                              // [4 128 32 32]
		ref2 := upsample(z1, x1)                                     // upsample to 64x64
		decode2 := decodeLayer(p.Sub("decode2"), 64+128, 64, ref2)   //
		z2 := x1.ApplyT(decode2, train)                              // [4 64 64 64]
		// ref3 := upsample(z2, x0)
		decode3 := decodeLayer(p.Sub("decode3"), 64+64, 32, z2) // z2 and x0 have same HxW = 64x64
		z3 := x0.ApplyT(decode3, train)                         // [4 32 64 64]
		ref4 := upsample(z3, x)                                 // upsample to 256x256
		decode4 := decodeLayer(p.Sub("decode4"), 32, 16, nil)   //
		z4 := ref4.ApplyT(decode4, train)                       // [4 16 256 256]
		logit := conv2d(p.Sub("logit"), 16, 1, 3, 1, 1)         // down sample to 1
		retVal := z4.ApplyT(logit, train)                       // [4  1 256 256]

		// Free memory
		x0.MustDrop()
		x1.MustDrop()
		x2.MustDrop()
		x3.MustDrop()
		x4.MustDrop()
		z.MustDrop()
		z0.MustDrop()
		z1.MustDrop()
		z2.MustDrop()
		z3.MustDrop()
		z4.MustDrop()

		ref0.MustDrop()
		ref1.MustDrop()
		ref2.MustDrop()
		// ref3.MustDrop()
		ref4.MustDrop()

		return retVal
	})
}

func runCheckModel() {
	// Create the model and load the weights from the file.
	modelPath, err := filepath.Abs(ModelPath)
	if err != nil {
		log.Fatal(err)
	}

	vs := nn.NewVarStore(Device)
	net := ResNet34Unet(vs.Root(), true)

	missings, err := vs.LoadPartial(modelPath)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Num of missings: %v\n", len(missings))
	for _, m := range missings {
		fmt.Printf("Missing Var: %v\n", m)
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
	// fmt.Printf("image: %i\n", image)
	si := CPUInfo()
	for i := 0; i < 100; i++ {
		ts.NoGrad(func() {
			si = CPUInfo()
			ram0 := si.TotalRam - si.FreeRam
			logit := net.ForwardT(image, false).MustTotype(gotch.Double, true)
			// loss := criterionBinaryCrossEntropy(logit, mask)
			// fmt.Printf("mask: %v\n", mask.MustSize())
			// fmt.Printf("image: %v\n", image.MustSize())
			// fmt.Printf("logit: %v\n", logit.MustSize())
			// l := loss.Float64Values()[0]
			logit.MustDrop()
			// loss.MustDrop()
			si = CPUInfo()
			ram1 := si.TotalRam - si.FreeRam
			fmt.Printf("%02d- Leak: %8.2fMB\n", i, float64(ram1-ram0)/1024)
			// fmt.Printf("%02d-Leak: %6.2fMB\t Loss: %v\n", i, float64(ram1-ram0)/1024, l)
		})
	}
}
