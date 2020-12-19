package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/chai2010/tiff"
	"github.com/disintegration/imaging"
	// gtiff "github.com/google/tiff"
	// gimage "github.com/google/tiff/image"
	// libtiff "github.com/andviro/go-libtiff/libtiff"
	// gotiff "golang.org/x/image/tiff"

	"github.com/nfnt/resize"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
	"golang.org/x/image/draw"
)

// readImage reads image from file.
func readImage(filename string) (image.Image, error) {
	ext := filepath.Ext(filename)
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	switch ext {
	case ".png", ".PNG":
		return png.Decode(f)
	case ".jpg", ".jpeg", ".JPG", ".JPEG":
		return jpeg.Decode(f)
	case ".tiff", ".tif", ".TIFF", ".TIF":
		return tiff.Decode(f)
	default:
		err = fmt.Errorf("Unsupported image format: %v\n", ext)
		return nil, err
	}
}

// splitTensor split image tensor(CxHxW) into tile images with specified size and offset
func splitTensor(img *ts.Tensor, imgName string, tileSize, offset int64) {
	size := img.MustSize()
	h, w := size[1], size[2]

	var dimY int64 = 0
	var dimX int64 = 1

	ys := int(math.Ceil(float64(h / offset)))
	xs := int(math.Ceil(float64(w / offset)))

	var offsetY int64 = 0
	for n := 0; n < ys; n++ {
		nextOffsetY := offsetY + tileSize
		narrowY := img.MustNarrow(dimY, offsetY, nextOffsetY, false).MustDetach(false)
		offsetY = nextOffsetY
		var offsetX int64 = 0
		for m := 0; m < xs; m++ {
			start := offsetX
			end := offsetX + tileSize
			tile := narrowY.MustNarrow(dimX, start, end, false).MustDetach(false)
			vision.Save(tile, fmt.Sprintf("%v_%v_%v.png", imgName, n, m))
			tile.MustDrop()
		}
	}
}

// isBlank checks whether image is black or gray based on saturation threshold
//
// NOTE. use saturation only because pixel counts are not consistent.
func isBlank(img image.Image) bool {
	var satThreshold float64 = 0.4
	pixelThreshold := 200

	maxY := img.Bounds().Size().Y
	maxX := img.Bounds().Size().X
	rec := image.Rectangle{image.ZP, image.Point{maxX, maxY}}
	imgSrc := image.NewNRGBA(rec)
	draw.Copy(imgSrc, image.ZP, img, img.Bounds(), draw.Src, nil)
	var satSum int = 0 // sum of pixels with saturation > threshold
	for y := 0; y < maxY; y++ {
		for x := 0; x < maxX; x++ {
			c := imgSrc.NRGBAAt(x, y)
			r, g, b := c.R, c.G, c.B
			s := Saturation(float64(r), float64(g), float64(b))
			if s > satThreshold {
				satSum += 1
			}
		}
	}
	/*
	 *   var pixels uint8 = 0
	 *   for _, p := range imgSrc.Pix {
	 *     pixels += p
	 *   }
	 *  */
	// fmt.Printf("Saturation sum: %v - pixels: %v\n", graySum, pixels)

	// gray or black
	// if graySum <= pixelThreshold || int(pixels) <= pixelThreshold {
	if satSum <= pixelThreshold {
		return true
	}

	return false
}

// toGrayScale converts RGB values to gray scale.
//
// ref: https://stackoverflow.com/questions/42516203
func toGrayScale(r, g, b float64) float64 {
	// luminosity
	lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
	return lum / 256
}

// Saturation calculates intensity of a hue.
//
// Ref.
// - https://en.wikipedia.org/wiki/Colorfulness
// - http://changingminds.org/explanations/perception/visual/hsl.htm
func Saturation(r, g, b float64) float64 {
	r = r / 255
	g = g / 255
	b = b / 255

	max := math.Max(math.Max(r, g), b)
	min := math.Min(math.Min(r, g), b)

	if max == 0 {
		return 0
	}

	lum := max - min

	if lum < 0.5 {
		return (max - min) / (max + min)
	} else {
		return (max - min) / (2 - max - min)
	}
}

// masking masks image with given mask.
func masking(img, maskImg string, shape []int64) error {
	imgExt := filepath.Ext(img)
	maskExt := filepath.Ext(maskImg)
	w := int(shape[0])
	h := int(shape[1])

	imgFile, err := os.Open(img)
	if err != nil {
		return err
	}
	defer imgFile.Close()

	var imgBuf image.Image
	switch imgExt {
	case ".tiff":
		if imgBuf, err = tiff.Decode(imgFile); err != nil {
			return err
		}
	case ".png":
		if imgBuf, err = png.Decode(imgFile); err != nil {
			return err
		}
	default:
		err = fmt.Errorf("Unsupported image type: %v", imgExt)
		return err
	}

	maskFile, err := os.Open(maskImg)
	if err != nil {
		return err
	}
	defer maskFile.Close()

	var maskBuf image.Image
	switch maskExt {
	case ".tiff":
		if maskBuf, err = tiff.Decode(maskFile); err != nil {
			return err
		}
	case ".png":
		if maskBuf, err = png.Decode(maskFile); err != nil {
			return err
		}
	default:
		err = fmt.Errorf("Unsupported image type: %v", imgExt)
		return err
	}

	rec := image.Rectangle{image.Point{0, 0}, image.Point{w, h}}
	dstImg := image.NewRGBA(rec)
	draw.Draw(dstImg, rec, imgBuf, image.Point{}, draw.Src)

	mask := image.NewUniform(color.Alpha{64}) // 25% opacity
	draw.DrawMask(dstImg, rec, maskBuf, image.Point{}, mask, image.Point{}, draw.Over)

	out, err := os.Create("image-mask.png")
	if err != nil {
		return err
	}
	err = png.Encode(out, dstImg)
	if err != nil {
		return err
	}

	return nil
}

// tiff2Png converts .tiff file to .png
func tiff2Png(filePath string) error {
	f, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	img, err := tiff.Decode(f)
	if err != nil {
		return err
	}

	ext := filepath.Ext(filePath)
	pngFilePath := filePath[:len(filePath)-len(ext)]

	pngFile, err := os.Create(pngFilePath)
	if err != nil {
		return err
	}

	return png.Encode(pngFile, img)
}

// runTile slices images into tiles and saves to png
func runTile(sampleId string, rle []int, reduced, tileW, tileH int) error {
	// Create folders for tiled images and tiled masks
	tileImgPath := fmt.Sprintf("%v/tile/image", DataPath)
	tileMaskPath := fmt.Sprintf("%v/tile/mask", DataPath)
	if _, err := os.Stat(tileMaskPath); os.IsNotExist(err) {
		err = os.MkdirAll(tileMaskPath, 0755)
		if err != nil {
			return err
		}
	}
	if _, err := os.Stat(tileImgPath); os.IsNotExist(err) {
		err = os.MkdirAll(tileImgPath, 0755)
		if err != nil {
			return err
		}
	}

	trainPath := fmt.Sprintf("%v/hubmap-kidney-segmentation/train", DataPath)
	imageFile := fmt.Sprintf("%v/%v.tiff", trainPath, sampleId)
	iReader, err := os.Open(imageFile)
	if err != nil {
		return err
	}
	defer iReader.Close()

	var img image.Image
	if img, err = tiff.Decode(iReader); err != nil {
		return err
	}

	// Create mask file
	shape := []int64{int64(img.Bounds().Max.X), int64(img.Bounds().Max.Y)}
	fmt.Printf("rle length: %v - shape: %v\n", len(rle), shape)
	err = rle2Mask(rle, shape, sampleId)
	if err != nil {
		return err
	}

	// mask is always png
	maskPath := fmt.Sprintf("%v/mask", DataPath)
	maskFile := fmt.Sprintf("%v/%v.png", maskPath, sampleId)
	mReader, err := os.Open(maskFile)
	if err != nil {
		return err
	}
	defer mReader.Close()

	mask, err := png.Decode(mReader)
	if err != nil {
		return err
	}

	// Reduce image by x4 times for prototyping
	if reduced > 1 {
		reducedW := img.Bounds().Max.X / Reduction
		reducedH := img.Bounds().Max.Y / Reduction
		img = resize.Resize(uint(reducedW), uint(reducedH), img, resize.Lanczos3)
		mask = resize.Resize(uint(reducedW), uint(reducedH), mask, resize.Lanczos3)
	}

	w := img.Bounds().Max.X
	h := img.Bounds().Max.Y

	cols := int(math.Ceil(float64(w) / float64(tileW)))
	rows := int(math.Ceil(float64(h) / float64(tileH)))
	maxX := img.Bounds().Max.X
	maxY := img.Bounds().Max.Y

	count := 1
	for n := 0; n < rows; n++ {
		startY := n * tileH
		if startY >= maxY {
			continue
		}

		endY := startY + tileH
		if endY > img.Bounds().Max.Y {
			endY = img.Bounds().Max.Y
		}

		for m := 0; m < cols; m++ {
			startX := m * tileW
			endX := startX + tileW
			if startX >= maxX {
				continue
			}
			if endX > img.Bounds().Max.X {
				endX = img.Bounds().Max.X
			}
			startPoint := image.Point{startX, startY}
			endPoint := image.Point{endX, endY}
			rec := image.Rectangle{startPoint, endPoint}

			subImg, err := cropImage(img, rec)
			if err != nil {
				return err
			}

			subMask, err := cropImage(mask, rec)
			if err != nil {
				return err
			}

			// If gray or black image, skip it
			if isBlank(subImg) {
				continue
			}

			outImage := fmt.Sprintf("%v/%v_%03d.png", tileImgPath, sampleId, count)
			pngImage, err := os.Create(outImage)
			if err != nil {
				return err
			}
			err = png.Encode(pngImage, subImg)
			if err != nil {
				return err
			}
			pngImage.Close()

			outMask := fmt.Sprintf("%v/%v_%03d.png", tileMaskPath, sampleId, count)
			pngMask, err := os.Create(outMask)
			if err != nil {
				return err
			}
			err = png.Encode(pngMask, subMask)
			if err != nil {
				return err
			}
			pngMask.Close()

			count += 1
		}
	}

	return nil
}

// cropImage takes an image and crops it to the specified rectangle.
func cropImage(img image.Image, crop image.Rectangle) (image.Image, error) {
	type subImager interface {
		SubImage(r image.Rectangle) image.Image
	}

	// img is an Image interface. This checks if the underlying value has a
	// method called SubImage. If it does, then we can use SubImage to crop the
	// image.
	simg, ok := img.(subImager)
	if !ok {
		return nil, fmt.Errorf("image does not support cropping")
	}

	return simg.SubImage(crop), nil
}

// pixSum sum pixels in a image
func pixSum(img image.Image) uint8 {
	bounds := img.Bounds()
	imgSrc := image.NewRGBA(bounds)
	draw.Draw(imgSrc, bounds, img, image.Point{}, draw.Src)
	var pixels uint8
	for _, p := range imgSrc.Pix {
		pixels += p
	}

	return pixels
}

func isOpaque(im image.Image) bool {
	// Check if image has Opaque() method:
	if oim, ok := im.(interface {
		isOpaque() bool
	}); ok {
		return oim.isOpaque() // It does, call it and return its result!
	}

	// No Opaque() method, we need to loop through all pixels and check manually:
	rect := im.Bounds()
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			if _, _, _, a := im.At(x, y).RGBA(); a != 0xffff {
				return false // Found a non-opaque pixel: image is non-opaque
			}
		}

	}
	return true // All pixels are opaque, so is the image
}

// ProcessImage creates image mask from database and tiles images and their masks
func processImage() {
	start := time.Now()

	// sampleId := "0486052bb"
	trainDataFile := fmt.Sprintf("%v/train.csv", DataPath)
	rleMap, err := readRLE(trainDataFile)
	if err != nil {
		log.Fatal(err)
	}

	for sampleId := range rleMap {
		// NOTE. skip this bigtiff as can't read with Go atm.
		if sampleId != "095bf7a1f" {
			// continue
			err = runTile(sampleId, rleMap[sampleId], Reduction, TileSize, TileSize)
			if err != nil {
				err := fmt.Errorf("Processing sampleId %v error: %v\n", sampleId, err)
				log.Fatal(err)
			}
		} else {
			continue
			/*
			 *       err := testTile(sampleId, rleMap[sampleId], Reduction, TileSize, TileSize)
			 *       if err != nil {
			 *         err := fmt.Errorf("Processing sampleId %v error: %v\n", sampleId, err)
			 *         return err
			 *       }
			 *  */
		}

		fmt.Printf("Processing %v...", sampleId)
		fmt.Printf("Completed\n")
	}

	fmt.Println("Image processing: completed.")
	fmt.Printf("Duration: %.2f (min)\n", time.Since(start).Minutes())
}

func testReadTiff() {
	files := []string{
		// "0486052bb.tiff",
		"095bf7a1f.tiff",
		// "1e2425f28.tiff",
		// "2f6ecfcdf.tiff",
		// "54f2eec69.tiff",
		// "aaa6a05cc.tiff",
		// "cb2d976f4.tiff",
		// "e79de561c.tiff",
	}

	for _, f := range files {

		fname := fmt.Sprintf("%v/hubmap-kidney-segmentation/train/%v", DataPath, f)
		r, err := os.Open(fname)
		if err != nil {
			panic(err)
		}

		// _, err = gimage.Decode(r)
		// _, _, err = tiff.DecodeAll(r)
		// _, err = gotiff.Decode(r)
		// _, err := libtiff.Open(fname)
		// config, err := tiff.DecodeConfig(r)
		// if err != nil {
		// err := fmt.Errorf("file %v: %v", f, err)
		// panic(err)
		// }
		// fmt.Printf("%v - size(HxW): %v x %v (pixels)\n", f, config.Height, config.Width)
		_, err = imaging.Decode(r)
		if err != nil {
			panic(err)
		}
	}
}

func testTile(sampleId string, rle []int, reduced, tileW, tileH int) error {
	// Create folders for tiled images and tiled masks
	tileImgPath := fmt.Sprintf("%v/tile/image", DataPath)
	tileMaskPath := fmt.Sprintf("%v/tile/mask", DataPath)
	if _, err := os.Stat(tileMaskPath); os.IsNotExist(err) {
		err = os.MkdirAll(tileMaskPath, 0755)
		if err != nil {
			return err
		}
	}
	if _, err := os.Stat(tileImgPath); os.IsNotExist(err) {
		err = os.MkdirAll(tileImgPath, 0755)
		if err != nil {
			return err
		}
	}

	trainPath := fmt.Sprintf("%v/train", DataPath)
	imageFile := fmt.Sprintf("%v/%v.tiff", trainPath, sampleId)
	iReader, err := os.Open(imageFile)
	if err != nil {
		return err
	}
	defer iReader.Close()

	var img image.Image
	if img, err = tiff.Decode(iReader); err != nil {
		return err
	}

	// Create mask file
	shape := []int64{int64(img.Bounds().Max.X), int64(img.Bounds().Max.Y)}
	fmt.Printf("rle length: %v - shape: %v\n", len(rle), shape)
	err = rle2Mask(rle, shape, sampleId)
	if err != nil {
		return err
	}

	// mask is always png
	maskPath := fmt.Sprintf("%v/mask", DataPath)
	maskFile := fmt.Sprintf("%v/%v.png", maskPath, sampleId)
	mReader, err := os.Open(maskFile)
	if err != nil {
		return err
	}
	defer mReader.Close()

	mask, err := png.Decode(mReader)
	if err != nil {
		return err
	}

	// Reduce image by x4 times for prototyping
	if reduced > 1 {
		reducedW := img.Bounds().Max.X / Reduction
		reducedH := img.Bounds().Max.Y / Reduction
		img = resize.Resize(uint(reducedW), uint(reducedH), img, resize.Lanczos3)
		mask = resize.Resize(uint(reducedW), uint(reducedH), mask, resize.Lanczos3)
	}

	w := img.Bounds().Max.X
	h := img.Bounds().Max.Y

	cols := int(math.Ceil(float64(w) / float64(tileW)))
	rows := int(math.Ceil(float64(h) / float64(tileH)))
	maxX := img.Bounds().Max.X
	maxY := img.Bounds().Max.Y

	count := 1
	for n := 0; n < rows; n++ {
		startY := n * tileH
		if startY >= maxY {
			continue
		}

		endY := startY + tileH
		if endY > img.Bounds().Max.Y {
			endY = img.Bounds().Max.Y
		}

		for m := 0; m < cols; m++ {
			startX := m * tileW
			endX := startX + tileW
			if startX >= maxX {
				continue
			}
			if endX > img.Bounds().Max.X {
				endX = img.Bounds().Max.X
			}
			startPoint := image.Point{startX, startY}
			endPoint := image.Point{endX, endY}
			rec := image.Rectangle{startPoint, endPoint}

			subImg, err := cropImage(img, rec)
			if err != nil {
				return err
			}

			subMask, err := cropImage(mask, rec)
			if err != nil {
				return err
			}

			/* // If gray or black image, skip it
			 * if isBlank(subImg) {
			 *   continue
			 * } */

			testImgPath := fmt.Sprintf("%v/test/image", DataPath)
			outImage := fmt.Sprintf("%v/%v_%03d.png", testImgPath, sampleId, count)
			pngImage, err := os.Create(outImage)
			if err != nil {
				return err
			}
			err = png.Encode(pngImage, subImg)
			if err != nil {
				return err
			}
			pngImage.Close()

			testMaskPath := fmt.Sprintf("%v/test/mask", DataPath)
			outMask := fmt.Sprintf("%v/%v_%03d.png", testMaskPath, sampleId, count)
			pngMask, err := os.Create(outMask)
			if err != nil {
				return err
			}
			err = png.Encode(pngMask, subMask)
			if err != nil {
				return err
			}
			pngMask.Close()

			count += 1
		}
	}

	return nil
}

// rgb2GrayScale converts a RGB (3xHxW) to grayscale image (HxW).
// ref. https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py#L196-L234
// (0.2989 * r + 0.587 * g + 0.114 * b)
func rgb2GrayScale(x *ts.Tensor) (*ts.Tensor, error) {
	size := x.MustSize()
	if len(size) < 3 {
		err := fmt.Errorf("Expect at least 3D tensor. Got %v dimensions.\n", len(size))
		return nil, err
	}

	// e.g [4, 3, 256, 256]
	chanSize := size[len(size)-3]
	if chanSize != 3 {
		err := fmt.Errorf("Expect image of 3 channels for RGB. Got %v .\n", chanSize)
		return nil, err
	}

	channels := x.MustUnbind(-3, false)
	r := channels[0].MustMul1(ts.FloatScalar(0.2989), true)
	g := channels[1].MustMul1(ts.FloatScalar(0.587), true)
	b := channels[2].MustMul1(ts.FloatScalar(0.114), true)

	rg := r.MustAdd(g, true)
	g.MustDrop()
	gray := rg.MustAdd(b, true)
	b.MustDrop()

	return gray, nil
}

func checkMask() {
	// fpath := "../../kaggle-hubmap/hubmap-resnet34-unet/data/tile/0.25_480_240_train/aaa6a05cc/y00000484_x00001501.mask.png"
	fpath := "../../kaggle-hubmap/hubmap-resnet34-unet/data/tile/0.25_480_240_train/aaa6a05cc/y00000484_x00001501.png"
	x, err := vision.Load(fpath)
	if err != nil {
		panic(err)
	}

	fmt.Printf("mask shape: %v\n", x.MustSize())

	gray, err := rgb2GrayScale(x)
	if err != nil {
		panic(err)
	}

	err = vision.Save(gray.MustUnsqueeze(0, true), "test-grayscale.png")
	if err != nil {
		panic(err)
	}
}
