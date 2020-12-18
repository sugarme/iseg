package main

import (
	"fmt"
	"log"
	"reflect"

	"github.com/jarrahkula/kaggle-hubmap/dutil"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

// HubmapDataset implement dutil.Dataset
type HubmapDataset struct {
	// kfold   KFold
	fnames []string
}

func NewHubmapDataset(fnames []string) *HubmapDataset {
	return &HubmapDataset{fnames: fnames}
}

func (ds *HubmapDataset) Len() int {
	return len(ds.fnames)
}

type ImageMask struct {
	image ts.Tensor
	mask  ts.Tensor
}

// Item implements Dataset interface
func (ds *HubmapDataset) Item(idx int) (interface{}, error) {
	fname := ds.fnames[idx]
	imgPath := fmt.Sprintf("%v/tile/image/%v", DataPath, fname)
	maskPath := fmt.Sprintf("%v/tile/mask/%v", DataPath, fname)

	imgTs, err := vision.Load(imgPath)
	if err != nil {
		return nil, err
	}
	img := imgTs.MustDiv1(ts.FloatScalar(255.0), true)

	maskTs, err := vision.Load(maskPath)
	if err != nil {
		return nil, err
	}

	maskGray, err := rgb2GrayScale(maskTs)
	if err != nil {
		return nil, err
	}
	maskTs.MustDrop()
	mask := maskGray.MustDiv1(ts.FloatScalar(255.0), true)

	return ImageMask{
		image: *img,
		mask:  *mask,
	}, err
}

func (ds *HubmapDataset) DType() reflect.Type {
	return reflect.TypeOf(ds.fnames)
}

func runCheckDataLoader() {
	// mockup
	mockName := "0486052bb"
	var fnames []string
	for i := 0; i < 347; i++ {
		n := fmt.Sprintf("%v_%03d", mockName, i+1)
		fnames = append(fnames, n)
	}

	batchSize := 16
	ds := NewHubmapDataset(fnames)
	s, err := dutil.NewBatchSampler(ds.Len(), batchSize, true, true)
	if err != nil {
		log.Fatal(err)
	}
	dl, err := dutil.NewDataLoader(ds, s)
	if err != nil {
		log.Fatal(err)
	}

	count := 0
	for dl.HasNext() {
		s, err := dl.Next()
		if err != nil {
			log.Fatal(err)
		}

		count++
		var (
			img, mask []ts.Tensor
		)

		for _, i := range s.([]ImageMask) {
			img = append(img, i.image)
			mask = append(mask, i.mask)
		}

		imgTs := ts.MustStack(img, 0)
		maskTs := ts.MustStack(mask, 0)

		// fmt.Printf("Loaded %v: %v, image shape: %v, mask shape: %v\n", count, len(s.([]ImageMask)), s.([]ImageMask)[0].image.MustSize(), s.([]ImageMask)[0].mask.MustSize())
		fmt.Printf("Loaded %v: %v, image shape: %v, mask shape: %v\n", count, len(s.([]ImageMask)), imgTs.MustSize(), maskTs.MustSize())
	}
}
