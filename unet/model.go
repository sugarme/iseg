package unet

import (
	// "fmt"

	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/iseg/encoder"
)

// Unet is a UNET model struct
// Ref: https://arxiv.org/abs/1505.04597
type UNet struct {
	encoder encoder.Encoder
	decoder *UNetDecoder
}

// ForwardT implements ts.ModuleT for UNet struct.
func (n *UNet) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	features := n.encoder.ForwardAll(x, train)
	logit := n.decoder.ForwardFeatures(features, train)
	for _, f := range features {
		f.MustDrop()
	}

	return logit
}

// DefaultUNet creates UNet with default values.
// ResNet34 as encoder.
func DefaultUNet(p *nn.Path) *UNet {
	enc := encoder.NewResNet34Encoder(p)
	dec := NewUNetDecoder(p)

	return &UNet{
		encoder: enc,
		decoder: dec,
	}
}
