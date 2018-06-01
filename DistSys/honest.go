package main

import (
	// "math/rand"
	"encoding/binary"
	"fmt"
	"github.com/kniren/gota/dataframe"
	"github.com/sbinet/go-python"
	"gonum.org/v1/gonum/mat"
	"math"
	"strconv"
)

var (
	pyLogModule   *python.PyObject
	pyLogInitFunc *python.PyObject
	pyLogPrivFunc *python.PyObject
	pyNumFeatures *python.PyObject
	pyTestModule  *python.PyObject
	pyTestFunc    *python.PyObject
	pyTrainFunc   *python.PyObject
)

const (
	samples         = 10 // L
	sampleDeviation = 0.1
	batch_size      = 10
	epsilon         = 1.0
	datasetPath     = "../ML/data/"
	codePath        = "../ML/code"
	convThreshold   = 0.05
)

type Honest struct {
	id           int
	data         dataframe.DataFrame
	update       Update
	blockUpdates []Update
	bc           *Blockchain
}

func init() {

	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

// Load data and initialize chain

func (honest *Honest) initializeData(datasetName string, numberOfNodes int) {

	fullData := getData(datasetPath + datasetName + ".csv")

	honest.data = divideData(fullData, numberOfNodes)[honest.id]

	createCSVs(honest.data, datasetName, honest.id)

	honest.bc = NewBlockchain(honest.data.Ncol())

	pyInit(datasetName + strconv.Itoa(honest.id))

}

// check for Convergence by calling TestModel that invokes puython to compute train and test error 

func (honest *Honest) checkConvergence() bool {

	fmt.Println("Going into Python")

	trainError, _ := testModel(honest.bc.getLatestGradient(), "global")

	fmt.Printf("Train Error is %d in Iteration %d", trainError, honest.bc.blocks[len(honest.bc.blocks)-1].Data.Iteration)

	if trainError < convThreshold {
		return true
	}

	return false
}

// calculates update by calling oneGradientStep function that invokes python and passing latest global model from the chain to it.

func (honest *Honest) computeUpdate(iterationCount int, datasetName string) {
	prevGradient := honest.bc.getLatestGradient()
	deltas, err := oneGradientStep(prevGradient)
	check(err)
	honest.update = Update{Iteration: iterationCount, Delta: deltas}
	// fmt.Println(honest.update)
}

// Initialize the python stuff using go-python

func pyInit(datasetName string) {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString(codePath))

	pyLogModule = python.PyImport_ImportModule("logistic_model")
	pyTestModule = python.PyImport_ImportModule("logistic_model_test")
	fmt.Println(pyTestModule)

	pyLogInitFunc = pyLogModule.GetAttrString("init")
	pyLogPrivFunc = pyLogModule.GetAttrString("privateFun")
	fmt.Println(pyLogPrivFunc)
	pyTrainFunc = pyTestModule.GetAttrString("train_error")
	fmt.Println(pyTrainFunc)
	pyTestFunc = pyTestModule.GetAttrString("test_error")
	fmt.Println(pyTestFunc)

	pyNumFeatures = pyLogInitFunc.CallFunction(python.PyString_FromString(datasetName), python.PyFloat_FromDouble(epsilon))
	numFeatures := python.PyInt_AsLong(pyNumFeatures)

	fmt.Printf("Sucessfully pulled dataset. Features: %d\n", numFeatures)

}

// calculate the next update using the latest global model on the chain inovking python

func oneGradientStep(globalW []float64) ([]float64, error) {

	argArray := python.PyList_New(len(globalW))

	for i := 0; i < len(globalW); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(globalW[i]))
	}

	// Either use full GD or SGD here
	result := pyLogPrivFunc.CallFunction(python.PyInt_FromLong(1), argArray,
		python.PyInt_FromLong(batch_size))

	// Convert the resulting array to a go byte array
	pyByteArray := python.PyByteArray_FromObject(result)
	goByteArray := python.PyByteArray_AsBytes(pyByteArray)

	var goFloatArray []float64
	size := len(goByteArray) / 8

	for i := 0; i < size; i++ {
		currIndex := i * 8
		bits := binary.LittleEndian.Uint64(goByteArray[currIndex : currIndex+8])
		aFloat := math.Float64frombits(bits)
		goFloatArray = append(goFloatArray, aFloat)
	}
	return goFloatArray, nil
}

// add an update to the record of updates received for the current iteration

func (honest *Honest) addBlockUpdate(update Update) int {

	honest.blockUpdates = append(honest.blockUpdates, update)
	return len(honest.blockUpdates)
}

// creates a block from all the updates recorded.

func (honest *Honest) createBlock(iterationCount int) Block {

	pulledGradient := make([]float64, honest.data.Ncol())
	pulledGradient = honest.bc.getLatestGradient()
	updatedGradient := make([]float64, honest.data.Ncol())
	deltaM := mat.NewDense(1, honest.data.Ncol(), make([]float64, honest.data.Ncol()))
	pulledGradientM := mat.NewDense(1, honest.data.Ncol(), pulledGradient)

	// Update Aggregation
	for _, update := range honest.blockUpdates {
		deltaM = mat.NewDense(1, honest.data.Ncol(), update.Delta)
		pulledGradientM.Add(pulledGradientM, deltaM)
	}
	mat.Row(updatedGradient, 0, pulledGradientM)

	// TODO: Insert RONI
	bData := BlockData{iterationCount, updatedGradient, honest.blockUpdates}
	honest.bc.AddBlock(bData) 

	var newBlock Block
	newBlock = *(honest.bc.blocks[len(honest.bc.blocks)-1])

	return newBlock


}


// Empty the updates recorded at the start of each iteration

func (honest *Honest) flushUpdates(numberOfNodes int) {

	honest.blockUpdates = honest.blockUpdates[:0]
}

//Test the current global model. Determine training and test error to see if model has converged 

func testModel(weights []float64, node string) (float64, float64) {

	argArray := python.PyList_New(len(weights))

	for i := 0; i < len(weights); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(weights[i]))
	}


	pyTrainResult := pyTrainFunc.CallFunction(argArray)

	fmt.Println(pyTrainResult)

	trainErr := python.PyFloat_AsDouble(pyTrainResult)

	pyTestResult := pyTestFunc.CallFunction(argArray)

	fmt.Println(pyTestResult)

	testErr := python.PyFloat_AsDouble(pyTestResult)

	return trainErr, testErr

}

// Divide the dataset equally among the number of nodes

func divideData(data dataframe.DataFrame, numberOfNodes int) []dataframe.DataFrame {


	var dividedData []dataframe.DataFrame
	indexes := make([]int, 0)

	var stepsize int
	start := 0
	end := 0

	stepsize = data.Nrow() / numberOfNodes

	for i := 0; i < numberOfNodes; i++ {

		if i == numberOfNodes-1 {
			end = data.Nrow()
		} else {
			end = start + stepsize
		}

		for j := 0; j < (end - start); j++ {
			if i == 0 {
				indexes = append(indexes, start+j)
			} else {

				if j < len(indexes) {
					indexes[j] = start + j
				} else {
					indexes = append(indexes, start+j)
				}
			}

		}
		dividedData = append(dividedData, data.Subset(indexes))
		start = start + stepsize

	}

	return dividedData

}

// creates a CSV for your part of the data

func createCSVs(nodeData dataframe.DataFrame, datasetName string, nodeID int) {

	// create a CSV for your part of the dataset
	filename := datasetName + strconv.Itoa(nodeID) + ".csv"
	file, err := os.Create(datasetPath + filename)
	check(err)
	nodeData.WriteCSV(bufio.NewWriter(file))

}

// reads data into dataframe

func getData(filePath string) dataframe.DataFrame {

	f, err := os.Open(filePath)
	check(err)
	df := dataframe.ReadCSV(bufio.NewReader(f))
	return df

}

// Error - checking

func check(e error) {

	if e != nil {
		panic(e)
	}

}

