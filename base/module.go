package base

import (
	// "fmt"
	"log"
	"reflect"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

// Identity is a nn.Module placeholder.
// It forwards the input tensor as such.
type Identity struct{}

// Forward implement nn.Module for Identity struct
func (i *Identity) Forward(x *ts.Tensor) *ts.Tensor {
	return x.MustDetach(false)
}

// Forward implement nn.ModuleT for Identity struct.
func (i *Identity) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	return x.MustDetach(false)
}

// NewIdentity creates a new Identity struct.
func NewIdentity() *Identity {
	return &Identity{}
}

// SCSE is concurrent spatial and channel squeeze and excitement module.
// Ref. https://arxiv.org/abs/1808.08127
type SCSE struct {
	cSE *nn.SequentialT
	sSE *nn.SequentialT
}

// ForwardT implement ts.ModuleT for SCSE struct.
func (m *SCSE) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	// fmt.Printf("x shape: %v\n", x.MustSize())
	cse := m.cSE.ForwardT(x, train)
	sse := m.sSE.ForwardT(x, train)
	cmul := x.MustMul(cse, false)
	smul := x.MustMul(sse, false)
	res := cmul.MustAdd(smul, false)

	cse.MustDrop()
	sse.MustDrop()
	cmul.MustDrop()
	smul.MustDrop()

	return res
}

// NewSCSE creates new SCSE.
func NewSCSE(p *nn.Path, cIn int64, reductionOpt ...int64) *SCSE {
	var reduction int64 = 16
	if len(reductionOpt) > 0 {
		reduction = reductionOpt[0]
	}

	// Channel squeeze excite
	chanSeq := nn.SeqT()
	chanSeq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustAdaptiveAvgPool2d([]int64{1, 1}, false)
	}))
	chanSeq.Add(Conv2d(p.Sub("sqzconv1"), cIn, cIn/reduction, 1, 0, 1))
	chanSeq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))
	chanSeq.Add(Conv2d(p.Sub("sqzconv2"), cIn/reduction, cIn, 1, 0, 1))
	chanSeq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustSigmoid(false)
	}))

	// Spatial squeeze excite
	spatSeq := nn.SeqT()
	spatSeq.Add(Conv2d(p.Sub("spatconv"), cIn, 1, 1, 0, 1))
	chanSeq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustSigmoid(false)
	}))

	return &SCSE{
		cSE: chanSeq,
		sSE: spatSeq,
	}
}

type Attention struct {
	attn ts.ModuleT
}

func (a *Attention) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	return a.attn.ForwardT(x, train)
}

// NewAttention creates a new Attention.
func NewAttention(moduleOpt ...ts.ModuleT) *Attention {

	var attention ts.ModuleT = &Identity{}
	if len(moduleOpt) > 0 {
		attention = moduleOpt[0]
		// Only support SCSE struct.
		typ := reflect.Indirect(reflect.ValueOf(attention)).Type()
		if typ.Name() != "SCSE" {
			log.Fatalf("Unsupported module type. Only support SCSE module type. Got %v\n", typ.Name())
		}
	}

	return &Attention{attention}
}

// Conv2d creates Conv2D module.
func Conv2d(p *nn.Path, cIn, cOut, ksize, padding, stride int64) *nn.Conv2D {
	config := nn.DefaultConv2DConfig()
	config.Stride = []int64{stride, stride}
	config.Padding = []int64{padding, padding}

	return nn.NewConv2D(p, cIn, cOut, ksize, config)
}

// Conv2dNoBias creates Conv2D with no bias.
func Conv2dNoBias(p *nn.Path, cIn, cOut, ksize, padding, stride int64) *nn.Conv2D {
	config := nn.DefaultConv2DConfig()
	config.Bias = false
	config.Stride = []int64{stride, stride}
	config.Padding = []int64{padding, padding}

	return nn.NewConv2D(p, cIn, cOut, ksize, config)
}

// Conv2dRelu creates a SequentialT composing of Conv2D No bias and a ReLU activation.
func Conv2dRelu(p *nn.Path, cIn, cOut, ksize, padding, stride int64) *nn.SequentialT {
	bnConfig := nn.DefaultBatchNormConfig()
	bnConfig.Eps = 0.001
	seq := nn.SeqT()
	seq.Add(Conv2dNoBias(p.Sub("conv"), cIn, cOut, ksize, padding, stride))
	seq.Add(nn.BatchNorm2D(p.Sub("bn"), cOut, bnConfig))
	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	return seq
}
