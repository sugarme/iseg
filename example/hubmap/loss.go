package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// Binary Cross Entropy with Logits
func criterionBinaryCrossEntropy(logit, mask *ts.Tensor) *ts.Tensor {
	logitR := logit.MustReshape([]int64{-1}, false)
	maskR := mask.MustReshape([]int64{-1}, false)

	// NOTE: reduction: none = 0; mean = 1; sum = 2. Default=mean
	// ref. https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.binary_cross_entropy_with_logits
	retVal := logitR.MustBinaryCrossEntropyWithLogits(maskR, ts.NewTensor(), ts.NewTensor(), 1, true).MustView([]int64{-1}, true)
	maskR.MustDrop()
	return retVal
}

// Binary Cross Entropy Loss
func BCELoss(probability, mask *ts.Tensor) *ts.Tensor {
	p := probability.MustView([]int64{-1}, false)
	t := mask.MustView([]int64{-1}, false)

	// 1-p
	p1 := p.MustMul1(ts.FloatScalar(-1), false).MustAdd1(ts.FloatScalar(1), true)
	// 1-t
	t1 := t.MustMul1(ts.FloatScalar(-1), false).MustAdd1(ts.FloatScalar(1), true)

	pclip := p.MustClip(ts.FloatScalar(1e-6), ts.FloatScalar(1), false)
	logp := pclip.MustLog(true)
	p1clip := p1.MustClip(ts.FloatScalar(1e-6), ts.FloatScalar(1), true)
	logn := p1clip.MustLog(true)

	// t * logp
	tlogp := t.MustMul(logp, true)
	// (1-t)*logn
	t1logn := t1.MustMul(logn, true)

	loss := tlogp.MustAdd(t1logn, true)
	t1logn.MustDrop()

	lossMean := loss.MustMean(gotch.Double, true)

	return lossMean
}

// DiceLoss measures overlap between 2
// Ref. https://github.com/pytorch/pytorch/issues/1249
// http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf
// https://www.jeremyjordan.me/semantic-segmentation/#:~:text=Another%20popular%20loss%20function%20for,denotes%20perfect%20and%20complete%20overlap.
func DiceScore(probability, mask *ts.Tensor) float64 {
	// Flatten
	iflat := probability.MustView([]int64{-1}, false)
	tflat := mask.MustView([]int64{-1}, false)
	p := iflat.MustGt(ts.FloatScalar(0.5), true)
	t := tflat.MustGt(ts.FloatScalar(0.5), true)
	ptMul := p.MustMul(t, false)
	overlap := ptMul.MustSum(gotch.Double, true).Float64Values()[0]
	union := p.MustSum(gotch.Double, true).Float64Values()[0] + t.MustSum(gotch.Double, true).Float64Values()[0]

	dice := (2 * overlap) / (union + 0.001)

	return dice
}

// Ref. https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08
// Ref. https://www.kaggle.com/finlay/pytorch-fcn-resnet50-in-20-minute
func SoftDiceLoss(x, y *ts.Tensor) *ts.Tensor {
	dims := []int64{-2, -1}
	smooth := 1.0

	xyMul := x.MustMul(y, false)
	tp := xyMul.MustSum1(dims, false, gotch.Double, true)

	y1 := y.MustAdd1(ts.FloatScalar(-1), false)
	xy1Mul := y1.MustMul(x, true)
	fp := xy1Mul.MustSum1(dims, false, gotch.Double, true)

	x1 := x.MustAdd1(ts.FloatScalar(-1), false)
	x1yMul := x1.MustMul(y, true)
	fn := x1yMul.MustSum1(dims, false, gotch.Double, true)

	numerator := tp.MustMul1(ts.FloatScalar(2.0), false).MustAdd1(ts.FloatScalar(smooth), true)
	denominator := numerator.MustAdd(fp, false).MustAdd(fn, false)

	dc := numerator.MustDiv(denominator, true)

	tp.MustDrop()
	fp.MustDrop()
	fn.MustDrop()
	denominator.MustDrop()

	mean := dc.MustMean(gotch.Double, true)

	retVal := mean.MustMul1(ts.FloatScalar(-1), true).MustAdd1(ts.FloatScalar(1), true)
	return retVal
}

// Ref. https://www.kaggle.com/finlay/pytorch-fcn-resnet50-in-20-minute
func LossFunc(logit, mask *ts.Tensor) *ts.Tensor {
	bce := criterionBinaryCrossEntropy(logit, mask).MustMul1(ts.FloatScalar(0.8), true)
	prob := logit.MustSigmoid(false)
	dice := SoftDiceLoss(prob, mask).MustMul1(ts.FloatScalar(0.2), true)
	prob.MustDrop()

	retVal := bce.MustAdd(dice, true)
	dice.MustDrop()

	return retVal
}

// Accuracy calculates true positive and true negative.
func Accuracy(input, target *ts.Tensor) (tp, tn float64) {
	iflat := input.MustView([]int64{-1}, false)
	tflat := target.MustView([]int64{-1}, false)
	p := iflat.MustGt(ts.FloatScalar(0.5), true)
	t := tflat.MustGt(ts.FloatScalar(0.5), true)
	ptMul := p.MustMul(t, false)
	overlap := ptMul.MustSum(gotch.Double, true).Float64Values()[0]
	tSum := t.MustSum(gotch.Double, false).Float64Values()[0]
	tp = overlap / tSum

	p1 := p.MustAdd1(ts.FloatScalar(-1), false)
	t1 := t.MustAdd1(ts.FloatScalar(-1), false)
	p1Sum := p1.MustSum(gotch.Double, false)
	t1Sum := t1.MustSum(gotch.Double, false)
	pt1Mul := p1Sum.MustMul(t1Sum, false)
	pt1Sum := pt1Mul.MustSum(gotch.Double, true)
	tn = pt1Sum.Float64Values()[0] / t1Sum.Float64Values()[0]

	return tp, tn
}

func runCheckDiceScore() {

	input := ts.MustOfSlice([]float64{0.3, 0.4, 0.6})  // false, false, true
	target := ts.MustOfSlice([]float64{0.3, 0.4, 0.6}) // false, false, true

	// mul = false, false, true . overlap = 1
	// union = 1 + 1 = 2

	dice := DiceScore(input, target)
	tp, tn := Accuracy(input, target)

	fmt.Printf("input: %v\n", input)
	fmt.Printf("target: %v\n", input)
	fmt.Printf("Dice Score: %v\n", dice) // 0.999...
	fmt.Printf("tp: %v\n", tp)
	fmt.Printf("tn: %v\n", tn)
}
