package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func runEDA() {
	fname := fmt.Sprintf("%v/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv", DataPath)
	f, err := os.Open(fname)
	if err != nil {
		panic(err)
	}

	defer f.Close()

	df := dataframe.ReadCSV(f, dataframe.HasHeader(true))

	age := df.Col("age").Float()
	if err != nil {
		panic(err)
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	v := make(plotter.Values, len(age))
	for i := 0; i < len(age); i++ {
		v[i] = age[i]
	}

	h, err := plotter.NewHist(v, 5)
	p.Title.Text = "Age Histogram"
	p.Add(h)

	p.Save(4*vg.Inch, 4*vg.Inch, "age-histo.png")

	w, err := p.WriterTo(4*vg.Inch, 4*vg.Inch, "age-histo")
	if err != nil {
		panic(err)
	}

	buf := new(bytes.Buffer)
	_, err = w.WriteTo(buf)
	if err != nil {
		panic(err)
	}

	buf.Bytes()
}

// HubMap Image struct
type HMImage struct {
	Id    string
	Shape []int64
}

func readDataset() ([]HMImage, error) {
	fname := fmt.Sprintf("%v/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv", DataPath)
	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}

	df := dataframe.ReadCSV(f, dataframe.HasHeader(true)).Select([]string{"image_file", "width_pixels", "height_pixels"})
	ids := df.Col("image_file").Records()
	widths := df.Col("width_pixels").Records()
	heights := df.Col("height_pixels").Records()
	var dataset []HMImage

	for i := 0; i < len(ids); i++ {
		fname := ids[i]
		ext := filepath.Ext(fname)
		id := ids[i][:len(fname)-len(ext)]
		w, err := strconv.Atoi(widths[i])
		if err != nil {
			return nil, err
		}
		h, err := strconv.Atoi(heights[i])
		if err != nil {
			return nil, err
		}

		dataset = append(dataset, HMImage{id, []int64{int64(w), int64(h)}})
	}

	return dataset, nil
}
