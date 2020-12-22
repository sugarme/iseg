package unet

import (
	// "fmt"
	"log"
	"reflect"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"

	"github.com/sugarme/iseg/base"
)

type DecoderLayer struct {
	Conv1 *nn.SequentialT
	Attn1 *base.Attention
	Conv2 *nn.SequentialT
	Attn2 *base.Attention
}

// interpolation using `nearest` algorithm
func upsample(x, ref *ts.Tensor) *ts.Tensor {
	xSize := x.MustSize()
	refSize := ref.MustSize()
	if reflect.DeepEqual(xSize[2:], refSize[2:]) {
		return x.MustDetach(false)
	}

	return x.MustUpsampleNearest2d(refSize[2:], nil, nil, false)
}

// ForwardSkip forwads and upsamples input tensor.
func (d *DecoderLayer) ForwardSkip(x, skip *ts.Tensor, train bool) *ts.Tensor {
	var attn1 *ts.Tensor
	var cat *ts.Tensor
	if skip != nil {
		cat = ts.MustCat([]ts.Tensor{*x, *skip}, 1)
	} else {
		// cat = x.MustDetach(false)
		cat = ts.MustCat([]ts.Tensor{*x}, 1)
	}
	attn1 = d.Attn1.ForwardT(cat, train)
	cat.MustDrop()
	conv1 := d.Conv1.ForwardT(attn1, train)
	attn1.MustDrop()
	conv2 := d.Conv2.ForwardT(conv1, train)
	conv1.MustDrop()
	res := d.Attn2.ForwardT(conv2, train)
	conv2.MustDrop()

	return res
}

// NewDecoderLayer creates a DecoderLayer.
func NewDecoderLayer(p *nn.Path, cIn, skip, cOut int64) *DecoderLayer {

	// fmt.Printf("cIn + skip: %v + %v = %v\n", cIn, skip, cIn+skip)
	conv1 := base.Conv2dRelu(p.Sub("conv1"), cIn+skip, cOut, 3, 1, 1)
	attn1 := base.NewAttention(base.NewSCSE(p.Sub("attn1"), cIn+skip))
	// attn1 := base.NewAttention()
	conv2 := base.Conv2dRelu(p.Sub("conv2"), cOut, cOut, 3, 1, 1)
	attn2 := base.NewAttention(base.NewSCSE(p.Sub("attn2"), cOut))
	// attn2 := base.NewAttention()

	return &DecoderLayer{
		Conv1: conv1,
		Attn1: attn1,
		Conv2: conv2,
		Attn2: attn2,
	}
}

type CenterLayer struct {
	Conv1 *nn.SequentialT
	Conv2 *nn.SequentialT
}

// ForwardT implements ts.ModuleT interface for CenterLayer struct.
func (c *CenterLayer) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	c1 := c.Conv1.ForwardT(x, train)
	c2 := c.Conv2.ForwardT(c1, train)
	c1.MustDrop()

	return c2
}

// NewCenterLayer creates new CenterLayer.
// TODO. add optional kernel size, padding.
func NewCenterLayer(p *nn.Path, cIn, cOut int64) *CenterLayer {
	conv1 := base.Conv2dRelu(p.Sub("conv1"), cIn, cOut, 3, 1, 1)
	conv2 := base.Conv2dRelu(p.Sub("conv2"), cOut, cOut, 3, 1, 1)

	return &CenterLayer{conv1, conv2}
}

// UNetDecoder is Decoder struct for UNet model.
type UNetDecoder struct {
	center  *nn.SequentialT
	decode0 *DecoderLayer
	decode1 *DecoderLayer
	decode2 *DecoderLayer
	decode3 *DecoderLayer
	decode4 *DecoderLayer
	logit   ts.ModuleT
}

// NewUNetDecoder creates UNetDecoder.
func NewUNetDecoder(p *nn.Path) *UNetDecoder {
	center := base.Conv2dRelu(p.Sub("center"), 512, 512, 11, 5, 1)
	decode0 := NewDecoderLayer(p.Sub("decoder0"), 512, 256, 256)
	decode1 := NewDecoderLayer(p.Sub("decoder1"), 256, 128, 128)
	decode2 := NewDecoderLayer(p.Sub("decoder2"), 128, 64, 64)
	decode3 := NewDecoderLayer(p.Sub("decoder3"), 64, 64, 32)
	decode4 := NewDecoderLayer(p.Sub("decoder4"), 32, 0, 16)
	logit := base.Conv2d(p.Sub("logit"), 16, 1, 3, 1, 1)

	return &UNetDecoder{
		center:  center,
		decode0: decode0,
		decode1: decode1,
		decode2: decode2,
		decode3: decode3,
		decode4: decode4,
		logit:   logit,
	}
}

// Forward forwards through input features.
func (n *UNetDecoder) ForwardFeatures(features []*ts.Tensor, train bool) *ts.Tensor {
	if len(features) != 6 {
		log.Fatalf("Expected features of 6 tensors. Got %v\n", len(features))
	}

	// feat5: [bz 512 8 8]
	center := n.center.ForwardT(features[5], train)        // center [bz 512 8 8]
	skip0 := upsample(center, features[4])                 // feat4  [bz 256 16 16]
	z0 := n.decode0.ForwardSkip(features[4], skip0, train) // z0     [bz 256 16 16]
	skip1 := upsample(z0, features[3])                     // feat3  [bz 128 32 32]
	z1 := n.decode1.ForwardSkip(features[3], skip1, train) // z1     [bz 128 32 32]
	skip2 := upsample(z1, features[2])                     // feat2  [bz 64 64 64]
	z2 := n.decode2.ForwardSkip(features[2], skip2, train) // z2     [bz 64 64 64]
	skip3 := upsample(z2, features[1])                     // feat1  [bz 64 64 64]
	z3 := n.decode3.ForwardSkip(features[1], skip3, train) // z3     [bz 32 64 64]
	skip4 := upsample(z3, features[0])                     // feat0  [bz 3 256 256]
	z4 := n.decode4.ForwardSkip(skip4, nil, train)         // z4     [bz 16 256 256]
	logit := n.logit.ForwardT(z4, train)                   // logit  [bz 16 256 256]

	center.MustDrop()
	z0.MustDrop()
	z1.MustDrop()
	z2.MustDrop()
	z3.MustDrop()
	z4.MustDrop()
	skip0.MustDrop()
	skip1.MustDrop()
	skip2.MustDrop()
	skip3.MustDrop()
	skip4.MustDrop()

	return logit
}
