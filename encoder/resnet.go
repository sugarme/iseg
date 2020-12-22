package encoder

import (
	"fmt"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

type ResNetEncoder struct {
	layer0 ts.ModuleT
	layer1 ts.ModuleT
	layer2 ts.ModuleT
	layer3 ts.ModuleT
	layer4 ts.ModuleT
}

// ForwardAll implements Encoder interface for ResNetEncoder
func (e *ResNetEncoder) ForwardAll(x *ts.Tensor, train bool) []*ts.Tensor {
	xn := rgbNormalize(x)
	x0 := e.layer0.ForwardT(xn, train)
	x1 := e.layer1.ForwardT(x0, train)
	x2 := e.layer2.ForwardT(x1, train)
	x3 := e.layer3.ForwardT(x2, train)
	x4 := e.layer4.ForwardT(x3, train)

	return []*ts.Tensor{xn, x0, x1, x2, x3, x4}
}

func NewResNet34Encoder(p *nn.Path) *ResNetEncoder {
	return &ResNetEncoder{
		layer0: layerZero(p), // NOTE. `conv1` and `bn1` are at root of pretrained model
		layer1: basicLayer(p.Sub("layer1"), 64, 64, 1, 3),
		layer2: basicLayer(p.Sub("layer2"), 64, 128, 2, 4),
		layer3: basicLayer(p.Sub("layer3"), 128, 256, 2, 6),
		layer4: basicLayer(p.Sub("layer4"), 256, 512, 2, 3),
	}
}

func rgbNormalize(x *ts.Tensor) *ts.Tensor {
	meanVals := []float32{0.485, 0.456, 0.406} // image RGB mean
	sdVals := []float32{0.229, 0.224, 0.225}   // image RGB standard error

	mean := ts.MustOfSlice(meanVals).MustView([]int64{1, 3, 1, 1}, true)
	sd := ts.MustOfSlice(sdVals).MustView([]int64{1, 3, 1, 1}, true)

	// x = (x - mean)/sd
	n := x.MustSub(mean, false).MustDiv(sd, true)
	mean.MustDrop()
	sd.MustDrop()

	return n
}

func layerZero(p *nn.Path) ts.ModuleT {
	conv1 := conv2dNoBias(p.Sub("conv1"), 3, 64, 7, 3, 2)
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
	layer.Add(NewBasicBlock(path.Sub("0"), cIn, cOut, stride))
	for blockIndex := 1; blockIndex < int(cnt); blockIndex++ {
		layer.Add(NewBasicBlock(path.Sub(fmt.Sprint(blockIndex)), cOut, cOut, 1))
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

func downSample(path *nn.Path, cIn, cOut, stride int64) ts.ModuleT {
	if stride != 1 || cIn != cOut {
		seq := nn.SeqT()
		seq.Add(conv2dNoBias(path.Sub("0"), cIn, cOut, 1, 0, stride))
		seq.Add(nn.BatchNorm2D(path.Sub("1"), cOut, nn.DefaultBatchNormConfig()))

		return seq
	}
	return nn.SeqT()
}

type BasicBlock struct {
	Conv1      *nn.Conv2D
	Bn1        *nn.BatchNorm
	Conv2      *nn.Conv2D
	Bn2        *nn.BatchNorm
	Downsample ts.ModuleT
}

func NewBasicBlock(path *nn.Path, cIn, cOut, stride int64) *BasicBlock {
	conv1 := conv2dNoBias(path.Sub("conv1"), cIn, cOut, 3, 1, stride)
	bn1 := nn.BatchNorm2D(path.Sub("bn1"), cOut, nn.DefaultBatchNormConfig())
	conv2 := conv2dNoBias(path.Sub("conv2"), cOut, cOut, 3, 1, 1)
	bn2 := nn.BatchNorm2D(path.Sub("bn2"), cOut, nn.DefaultBatchNormConfig())
	downsample := downSample(path.Sub("downsample"), cIn, cOut, stride)

	return &BasicBlock{conv1, bn1, conv2, bn2, downsample}
}

func (bb *BasicBlock) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	c1 := bb.Conv1.ForwardT(x, train)
	bn1Ts := bb.Bn1.ForwardT(c1, train)
	c1.MustDrop()
	relu := bn1Ts.MustRelu(true)
	c2 := bb.Conv2.ForwardT(relu, train)
	relu.MustDrop()
	bn2Ts := bb.Bn2.ForwardT(c2, train)
	c2.MustDrop()
	dsl := bb.Downsample.ForwardT(x, train)
	dslAdd := dsl.MustAdd(bn2Ts, true)
	bn2Ts.MustDrop()
	res := dslAdd.MustRelu(true)

	return res
}
