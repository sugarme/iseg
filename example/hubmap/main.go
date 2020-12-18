package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"

	"github.com/sugarme/gotch"
)

// flag variables
var (
	DataPath  string
	OptStr    string
	ModelPath string
	Cuda      bool
	task      string
	Device    gotch.Device
)

// hyperparameters
var (
	Reduction    int     // image resolution reduction times
	TileSize     int     // image tile size
	LR           float64 // learning rate
	BatchSize    int     // batch size
	ValidateSize int     // number of iterations that triggers running validation task
)

func init() {
	flag.StringVar(&DataPath, "input", "./input", "specify input data directory")
	flag.StringVar(&ModelPath, "model", "./model/resnet34.ot", "specify full path to model weight '.ot' file.")
	flag.BoolVar(&Cuda, "cuda", false, "specify whether using CUDA or not.")
	flag.StringVar(&task, "task", "train", "specify task to run")
	flag.Float64Var(&LR, "lr", 0.001, "specify learning rate")
	flag.IntVar(&Reduction, "reduction", 4, "specify image resolution reduction times")
	flag.IntVar(&BatchSize, "batch", 16, "specify batch size")
	flag.IntVar(&ValidateSize, "validate", 10, "specify size of validation cycles.")
	flag.IntVar(&TileSize, "tile", 256, "specify tile image size")
	flag.StringVar(&OptStr, "opt", "SGD", "specify optimizer type")
}

func main() {
	flag.Parse()

	DataPath = absPath(DataPath)
	ModelPath = absPath(ModelPath)

	Device = gotch.CPU
	if Cuda {
		Device = gotch.NewCuda().CudaIfAvailable()
	}

	switch task {
	case "model":
		runCheckModel()
	case "train":
		runTrain()
	case "eda":
		runEDA()
	case "image":
		processImage()
	default:
		err := fmt.Errorf("Unknown 'task' name. Please specify valid 'task' flag to run.\n")
		panic(err)
	}
}

// helper to get absolute file path
func absPath(p string) string {
	fullpath, err := filepath.Abs(p)
	if err != nil {
		log.Fatal(err)
	}
	return fullpath
}
