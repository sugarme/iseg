package unet

import (
	// "fmt"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/iseg/base"
	"github.com/sugarme/iseg/encoder"
)

// Unet is a UNET model struct
// Ref: https://arxiv.org/abs/1505.04597
type UNet struct {
	encoder encoder.Encoder
	decoder *UNetDecoder
	segHead *nn.SequentialT
}

// ForwardT implements ts.ModuleT for UNet struct.
func (n *UNet) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	// 0- Shape: [36 3 256 256]
	// 1- Shape: [36 64 64 64]
	// 2- Shape: [36 64 64 64]
	// 3- Shape: [36 128 32 32]
	// 4- Shape: [36 256 16 16]
	// 5- Shape: [36 512 8 8]
	features := n.encoder.ForwardAll(x, train)
	out := n.decoder.ForwardFeatures(features, train)
	// fmt.Printf("out shape: %v\n", out.MustSize())
	segHead := n.segHead.ForwardT(out, train)
	masks := upsample(segHead, x)

	for _, f := range features {
		f.MustDrop()
	}
	out.MustDrop()
	segHead.MustDrop()

	return masks
}

// DefaultUNet creates UNet with default values.
// ResNet34 as encoder.
func DefaultUNet(p *nn.Path) *UNet {
	encoderChannels := []int64{3, 64, 64, 128, 256, 512}
	decoderChannels := []int64{256, 128, 64, 32, 16}
	enc := encoder.NewResNet34Encoder(p)
	dec := NewUNetDecoder(p, encoderChannels, decoderChannels, 5)

	// cIn=decoderChannels[-1], cOut=classes, ksize(kernel size = 3)
	head := base.NewSegmentationHead(p.Sub("logit"), 16, 1, 3)

	return &UNet{
		encoder: enc,
		decoder: dec,
		segHead: head,
	}
}
