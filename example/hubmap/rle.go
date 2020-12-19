package main

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/go-gota/gota/dataframe"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

// readRLE reads run-length encoding data from CSV file and returns map id to rle slice
func readRLE(filename string) (map[string][]int, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	df := dataframe.ReadCSV(f, dataframe.HasHeader(true))
	ids := df.Col("id").Records()

	var rleMap map[string][]int = make(map[string][]int, 0)
	re := regexp.MustCompile(`[0-9]`)

	for idx, id := range ids {
		rleStrings := strings.Split(df.Subset(idx).Col("encoding").String(), " ")

		var rle []int
		for _, s := range rleStrings {
			digits := re.FindAllString(s, -1)
			num, err := strconv.Atoi(strings.Join(digits, ""))
			if err != nil {
				return nil, err
			}

			rle = append(rle, num)
		}
		rleMap[id] = rle
	}

	return rleMap, nil
}

// rle2Mask converts run-length encoding to mask array
//
// rle: run-length encoding
// shape: mask array dimensions
// sampleId: sampleId corresponding to image file
func rle2Mask(rle []int, shape []int64, sampleId string) error {
	// Create mask folder if not existing
	maskPath := fmt.Sprintf("%v/mask", DataPath)
	if _, err := os.Stat(maskPath); os.IsNotExist(err) {
		err = os.MkdirAll(maskPath, 0755)
		if err != nil {
			return err
		}
	}

	width := shape[0]
	height := shape[1]

	var rlePairs [][]int
	for i := 0; i < len(rle); i += 2 {
		rlePairs = append(rlePairs, []int{rle[i], rle[i+1]})
	}

	var pixels []uint8
	for i := 0; i < int(width*height); i++ {
		pixels = append(pixels, 0)
	}

	for _, p := range rlePairs {
		start := p[0]
		end := start + p[1]

		for i := start; i < end; i++ {
			v := 255 // int
			pixels[i] = uint8(v)
		}
	}

	imgTs, err := ts.NewTensorFromData(pixels, shape)
	if err != nil {
		return err
	}

	img := imgTs.MustTranspose(0, 1, true).MustUnsqueeze(0, true)

	filePath := fmt.Sprintf("%v/%v.png", maskPath, sampleId)
	return vision.Save(img, filePath)
}
