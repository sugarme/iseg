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
		return x
	}

	return x.MustUpsampleNearest2d(refSize[2:], nil, nil, false)
}

// ForwardSkip forwads and upsamples input tensor.
func (d *DecoderLayer) ForwardSkip(x, skip *ts.Tensor, train bool) *ts.Tensor {
	var attn1 *ts.Tensor
	if skip != nil {
		xup := upsample(x, skip)
		/*
		 *     fmt.Printf("x size: %v\n", x.MustSize())
		 *     fmt.Printf("skip size: %v\n", skip.MustSize())
		 *     fmt.Printf("xup size: %v\n", xup.MustSize())
		 *  */
		xcat := ts.MustCat([]ts.Tensor{*xup, *skip}, 1)
		xup.MustDrop()
		attn1 = d.Attn1.ForwardT(xcat, train)
		xcat.MustDrop()
	} else {
		attn1 = x.MustDetach(false)
	}

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
	conv1 := base.Conv2dRelu(p, cIn+skip, cOut, 3, 1, 1)
	// attn1 := base.NewAttention(base.NewSCSE(p, cIn+skip))
	attn1 := base.NewAttention()
	conv2 := base.Conv2dRelu(p, cOut, cOut, 3, 1, 1)
	// attn2 := base.NewAttention(base.NewSCSE(p, cOut))
	attn2 := base.NewAttention()

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
	conv1 := base.Conv2dRelu(p, cIn, cOut, 3, 1, 1)
	conv2 := base.Conv2dRelu(p, cOut, cOut, 3, 1, 1)

	return &CenterLayer{conv1, conv2}
}

// UNetDecoder is Decoder struct for UNet model.
type UNetDecoder struct {
	center ts.ModuleT
	layers []*DecoderLayer
}

// Forward forwards through input features.
func (n *UNetDecoder) ForwardFeatures(features []*ts.Tensor, train bool) *ts.Tensor {
	// remove first skip as having same spatial resolution
	// and reverse channels to start from head fo encoder.
	var feats []*ts.Tensor
	for i := len(features) - 1; i >= 1; i-- {
		feats = append(feats, features[i])
	}

	head := feats[0]
	skips := feats[1:]
	/*
	 *   fmt.Printf("num of input features: %v\n", len(features))
	 *   fmt.Printf("num of layers: %v\n", len(n.layers))
	 *
	 *   fmt.Println("Inputs:")
	 *   for i, f := range features {
	 *     fmt.Printf("%v- Shape: %v\n", i, f.MustSize())
	 *   }
	 *
	 *   fmt.Printf("skips:\n")
	 *   for i, s := range skips {
	 *     fmt.Printf("%v- Shape: %v\n", i, s.MustSize())
	 *   }
	 *
	 *   fmt.Printf("feats:\n")
	 *   for i, f := range feats {
	 *     fmt.Printf("%v- Shape: %v\n", i, f.MustSize())
	 *   }
	 *  */
	x := n.center.ForwardT(head, train)
	var outs []*ts.Tensor = make([]*ts.Tensor, len(n.layers))
	outs[0] = x
	for i, l := range n.layers {
		switch {
		case i == 0:
			outs[1] = l.ForwardSkip(x, skips[0], train)
			// fmt.Printf("i: %v - cIn: %v - skip: %v - cOut: %v\n", i, outs[i].MustSize(), skips[i].MustSize(), outs[i+1].MustSize())
		case i == len(n.layers)-1: // last layer
			// fmt.Printf("i: %v - cIn: %v - skip: %v\n", i, outs[i].MustSize(), "NONE")
			retVal := l.ForwardSkip(outs[i], nil, train)
			// fmt.Printf("cOut: %v\n", retVal.MustSize())
			for _, o := range outs {
				o.MustDrop()
			}
			return retVal
		default:
			// fmt.Printf("i: %v - cIn: %v - skip: %v\n", i, outs[i].MustSize(), skips[i].MustSize())
			outs[i+1] = l.ForwardSkip(outs[i], skips[i], train)
			// fmt.Printf("cOut: %v\n", outs[i+1].MustSize())
		}
	}

	panic("Shouldn't reach here!!!")
}

// NewUNetDecoder creates UNetDecoder.
//
// Example:
// - encoderChannels: [  3,  64, 64, 128, 256, 512]
// - decoderChannels: 			[256, 128, 64,  32, 16]
// - numLayers: 			5
func NewUNetDecoder(p *nn.Path, encoderChannels, decoderChannels []int64, numLayers int) *UNetDecoder {
	if len(decoderChannels) != numLayers {
		log.Fatalf("Mismatched model depth (%v) and number of 'decoder channels' layers (%v)\n", numLayers, len(decoderChannels))
	}

	// remove first skip (as same spatial resolution) and reverse order
	var encoderChans []int64
	for i := len(encoderChannels) - 1; i >= 1; i-- {
		encoderChans = append(encoderChans, encoderChannels[i])
	}

	headChannels := encoderChans[0]
	var inChannels []int64
	inChannels = append(inChannels, headChannels)
	inChannels = append(inChannels, decoderChannels[:len(decoderChannels)-1]...)
	var skipChannels []int64
	skipChannels = append(skipChannels, encoderChans[1:]...)
	skipChannels = append(skipChannels, 0)
	outChannels := decoderChannels

	// center := NewCenterLayer(p, headChannels, headChannels)
	center := base.NewIdentity()

	var layers []*DecoderLayer
	for i := 0; i < len(inChannels); i++ {
		l := NewDecoderLayer(p, inChannels[i], skipChannels[i], outChannels[i])
		layers = append(layers, l)
	}

	return &UNetDecoder{
		center: center,
		layers: layers,
	}
}
