package base

import "github.com/sugarme/gotch/nn"

// NewSegmentationHead creates new SegmentatationHead (nn.SequentialT)
// TODO: add upsampling, activation func options.
func NewSegmentationHead(p *nn.Path, cIn, cOut, ksize int64) *nn.SequentialT {
	seq := nn.SeqT()
	seq.Add(Conv2d(p, cIn, cOut, ksize, 1, 1))
	// TODO:
	// if upsampling > 1 seq.Add(UpSamplingBilinear2d) else nn.Identity
	// seq.Add(ActivationFunc)

	return seq
}
