package unet

import (
	// "fmt"
	"reflect"

	// "github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"

	"github.com/sugarme/iseg/base"
)

// Down is a SequentialT module composed of maxpool and 2x conv.
type Down struct {
	MaxpoolConv *nn.SequentialT
}

// NewDown creates a new Down ModuleT layer.
func NewDown(p *nn.Path, cIn, cOut int64, cMidOpt ...int64) *Down {
	doubleconv := base.DoubleConv(p, cIn, cOut, cMidOpt...)

	down := nn.SeqT()
	down.AddFn(nn.NewFunc(func(x *ts.Tensor) *ts.Tensor {
		// Down sample to half size: [BCHW] => [B C/2 H/2 W/2]
		// ksize = 2; stride=2; padding=0; dilation=1; ceil=false
		down := x.MustMaxPool2d([]int64{2, 2}, []int64{2, 2}, []int64{0, 0}, []int64{1, 1}, false, false)
		return down
	}))
	down.Add(doubleconv)

	return &Down{down}
}

// ForwardT implements nn.ModuleT interface.
func (l *Down) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	return l.MaxpoolConv.ForwardT(x, train)
}

// Up is a SequentialT composed of an upsampling layer and a conv.
type Up struct {
	DoubleConv *nn.SequentialT
}

// NewUp creates new Up layer.
func NewUp(p *nn.Path, cIn, cOut int64) *Up {
	doubleconv := base.DoubleConv(p, cIn, cOut)
	return &Up{doubleconv}
}

// UpForward upsamples and forwards through double conv.
// x1, x2 should be in shape [Batch CHW]
func (l *Up) UpForward(x1, x2 *ts.Tensor, train bool) *ts.Tensor {
	// Upscaling
	// x1Size := x1.MustSize()
	x2Size := x2.MustSize()
	xUp := upsampling(x1, x2Size[2:])
	/*
	 *   // padding
	 *   // Ref. https://pytorch.org/docs/stable/nn.functional.html#pad
	 *   diffY := x2Size[2] - x1Size[2]
	 *   diffX := x2Size[3] - x1Size[3]
	 *   pad := []int64{diffX / 2, diffX - diffX/2, diffY / 2, diffY - diffY/2}
	 *
	 *   fmt.Printf("pad: %v\n", pad)
	 *   xPad := xUp.MustConstantPadNd(pad, true)
	 *   fmt.Printf("xPad: %v\n", xPad.MustSize())
	 *   fmt.Printf("x2: %v\n", x2.MustSize())
	 *  */
	// concatenating
	x := ts.MustCat([]ts.Tensor{*x2, *xUp}, 1)
	xUp.MustDrop()
	// xPad.MustDrop()

	// Forward through double conv
	out := l.DoubleConv.ForwardT(x, train)
	x.MustDrop()

	return out
}

// interpolation using `bilinear` algorithm
// x, ref should be in shape: [BatchSize CHW]
func upsampling(x *ts.Tensor, outSize []int64) *ts.Tensor {
	xSize := x.MustSize()
	if reflect.DeepEqual(xSize[2:], outSize) {
		return x.MustDetach(false)
	}

	return x.MustUpsampleBilinear2d(outSize, false, nil, nil, false)
}

// OutConv creates out layer.
func OutConv(p *nn.Path, cIn, cOut int64) *nn.Conv2D {
	config := nn.DefaultConv2DConfig()
	return nn.NewConv2D(p, cIn, cOut, 1, config)
}

// UNetOriginal is a UNet model.
type UNetOriginal struct {
	Inc *nn.SequentialT

	Down1 *Down
	Down2 *Down
	Down3 *Down
	Down4 *Down

	Up1 *Up
	Up2 *Up
	Up3 *Up
	Up4 *Up

	OutC *nn.Conv2D
}

// NewUNetOriginal creates a default UNet
// with 3 channels, 1 class, using bilinear mode.
func NewUNetOriginal(p *nn.Path) *UNetOriginal {
	inc := base.DoubleConv(p.Sub("inc"), 3, 64)
	down1 := NewDown(p.Sub("down1"), 64, 128)
	down2 := NewDown(p.Sub("down2"), 128, 256)
	down3 := NewDown(p.Sub("down3"), 256, 512)
	down4 := NewDown(p.Sub("down4"), 512, 1024/2) // bilinear: 1024/2

	up1 := NewUp(p.Sub("up1"), 1024, 512/2) // bilinear: 512/2
	up2 := NewUp(p.Sub("up2"), 512, 256/2)  // 128
	up3 := NewUp(p.Sub("up3"), 256, 128/2)  // 64
	up4 := NewUp(p.Sub("up4"), 128, 64)
	outc := OutConv(p.Sub("outc"), 64, 1)

	return &UNetOriginal{
		Inc:   inc,
		Down1: down1,
		Down2: down2,
		Down3: down3,
		Down4: down4,
		Up1:   up1,
		Up2:   up2,
		Up3:   up3,
		Up4:   up4,
		OutC:  outc,
	}
}

// ForwardT implements ts.ModuleT for UNet model
func (m *UNetOriginal) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	x1 := m.Inc.ForwardT(x, train)    // [B  64 H    W   ]
	x2 := m.Down1.ForwardT(x1, train) // [B 128 H/2  W/2 ]
	x3 := m.Down2.ForwardT(x2, train) // [B 256 H/4  W/4 ]
	x4 := m.Down3.ForwardT(x3, train) // [B 512 H/8  W/8 ]
	x5 := m.Down4.ForwardT(x4, train) // [B 512 H/16 W/16]

	z1 := m.Up1.UpForward(x5, x4, train) // [B 256 H/8 W/8]
	z2 := m.Up2.UpForward(z1, x3, train) // [B 128 H/4 W/4]
	z3 := m.Up3.UpForward(z2, x2, train) // [B  64 H/2 W/2]
	z4 := m.Up4.UpForward(z3, x1, train) // [B  64 H/1 W/1]

	logits := m.OutC.ForwardT(z4, train) // [B   1 H/1 W/1]

	x1.MustDrop()
	x2.MustDrop()
	x3.MustDrop()
	x4.MustDrop()
	x5.MustDrop()
	z1.MustDrop()
	z2.MustDrop()
	z3.MustDrop()
	z4.MustDrop()

	return logits
}
