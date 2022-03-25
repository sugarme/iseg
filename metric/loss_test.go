package metric_test

import (
	// "reflect"
	"fmt"
	"testing"

	"github.com/sugarme/gotch/ts"

	"github.com/sugarme/iseg/metric"
)

func TestDiceLoss(t *testing.T) {
	pslice := []int64{1, 0, 0, 1, 0, 0, 1, 0, 0}
	tslice := []int64{1, 0, 0, 1, 1, 0, 1, 0, 0}

	pred := ts.MustOfSlice(pslice).MustView([]int64{1, 3, 3}, true)
	target := ts.MustOfSlice(tslice).MustView([]int64{1, 3, 3}, true)

	loss := metric.DiceCoeff(pred, target)

	fmt.Printf("Dice Loss: %0.4f\n", loss)
}

func TestJaccardIndex(t *testing.T) {
	pslice := []int64{1, 0, 0, 1, 0, 0, 1, 0, 0}
	tslice := []int64{1, 0, 0, 1, 1, 0, 1, 0, 0}

	pred := ts.MustOfSlice(pslice).MustView([]int64{1, 3, 3}, true)
	target := ts.MustOfSlice(tslice).MustView([]int64{1, 3, 3}, true)

	iou := metric.JaccardIndex(pred, target, 2)
	fmt.Printf("IoU: %0.4f\n", iou)
}

func TestIoU(t *testing.T) {
	pslice := []int64{1, 0, 0, 1, 0, 0, 1, 0, 0}
	tslice := []int64{1, 0, 0, 1, 1, 0, 1, 0, 0}

	pred := ts.MustOfSlice(pslice).MustView([]int64{1, 3, 3}, true)
	target := ts.MustOfSlice(tslice).MustView([]int64{1, 3, 3}, true)

	iou := metric.IoU(pred, target)
	fmt.Printf("IoU: %0.4f\n", iou) // 0.7500
}

func TestDiceCoeff(t *testing.T) {
	pslice := []int64{1, 0, 0, 1, 0, 0, 1, 0, 0}
	tslice := []int64{1, 0, 0, 1, 1, 0, 1, 0, 0}

	pred := ts.MustOfSlice(pslice).MustView([]int64{1, 3, 3}, true)
	target := ts.MustOfSlice(tslice).MustView([]int64{1, 3, 3}, true)

	iou := metric.DiceCoeff(pred, target)
	fmt.Printf("IoU: %0.4f\n", iou) // 0.8571
}
