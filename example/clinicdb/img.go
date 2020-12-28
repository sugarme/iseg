package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"path/filepath"

	"github.com/chai2010/tiff"
	// "github.com/disintegration/imaging"
	// gtiff "github.com/google/tiff"
	// gimage "github.com/google/tiff/image"
	// libtiff "github.com/andviro/go-libtiff/libtiff"
	// gotiff "golang.org/x/image/tiff"
	// "github.com/nfnt/resize"
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
