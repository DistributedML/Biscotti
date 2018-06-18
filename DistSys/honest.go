package main

import (
	// "math/rand"
	"encoding/binary"
	"github.com/kniren/gota/dataframe"
	"github.com/sbinet/go-python"
	"gonum.org/v1/gonum/mat"
	"math"
	"strconv"
	"os"
	"bufio"
	"errors"
	"runtime"

)

var (
	pyLogModule   *python.PyObject
	pyLogInitFunc *python.PyObject
	pyLogPrivFunc *python.PyObject
	pyNumFeatures *python.PyObject
	pyTestModule  *python.PyObject
	pyTestFunc    *python.PyObject
	pyTrainFunc   *python.PyObject

	//Errors
	 blockExistsError = errors.New("Forbidden overwrite of block foiled")
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

	trainError, _ := testModel(honest.bc.getLatestGradient(), "global")

	outLog.Printf("Train Error is %d in Iteration %d", trainError, honest.bc.blocks[len(honest.bc.blocks)-1].Data.Iteration)

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
}

// Initialize the python stuff using go-python

func pyInit(datasetName string) {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString(codePath))

	pyLogModule = python.PyImport_ImportModule("logistic_model")
	pyTestModule = python.PyImport_ImportModule("logistic_model_test")

	pyLogInitFunc = pyLogModule.GetAttrString("init")
	pyLogPrivFunc = pyLogModule.GetAttrString("privateFun")
	pyTrainFunc = pyTestModule.GetAttrString("train_error")
	pyTestFunc = pyTestModule.GetAttrString("test_error")

	pyNumFeatures = pyLogInitFunc.CallFunction(python.PyString_FromString(datasetName), python.PyFloat_FromDouble(epsilon))
	numFeatures := python.PyInt_AsLong(pyNumFeatures)

	outLog.Printf("Sucessfully pulled dataset. Features: %d\n", numFeatures)

}

// calculate the next update using the latest global model on the chain invoking python

func oneGradientStep(globalW []float64) ([]float64, error) {

	runtime.LockOSThread()

	_gstate := python.PyGILState_Ensure()

	argArray := python.PyList_New(len(globalW))

	for i := 0; i < len(globalW); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(globalW[i]))
	}
	
	// Either use full GD or SGD here
	result := pyLogPrivFunc.CallFunction(python.PyInt_FromLong(1), argArray, python.PyInt_FromLong(batch_size))

	// Convert the resulting array to a go byte array
	pyByteArray := python.PyByteArray_FromObject(result)
	goByteArray := python.PyByteArray_AsBytes(pyByteArray)

	python.PyGILState_Release(_gstate)

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

// getIterationBlock


// add an update to the record of updates received for the current iteration

func (honest *Honest) addBlockUpdate(update Update) int {

	honest.blockUpdates = append(honest.blockUpdates, update)
	return len(honest.blockUpdates)
}

// creates a block from all the updates recorded.

func (honest *Honest) createBlock(iterationCount int) (*Block,error) {

	// Has block already been appended from advertisements by other client?
	if(honest.bc.getBlock(iterationCount) != nil){
		return nil, blockExistsError
	}


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

	newBlock := honest.bc.blocks[len(honest.bc.blocks)-1]

	return newBlock,nil


}

func (honest *Honest) addBlock(newBlock Block) error {

	// if already exists don't create/replace it
	if(honest.bc.getBlock(iterationCount) != nil){
		
		better := honest.evaluateBlockQuality(newBlock)
		if(!better){
			return blockExistsError
		}else{
			honest.replaceBlock(newBlock, newBlock.Data.Iteration) // this thing doesn't need the second argument I think
			return nil
		}
	
	}else{
	
		client.bc.AddBlockMsg(newBlock)
		return nil

	}
}

// Empty the updates recorded at the start of each iteration

func (honest *Honest) flushUpdates(numberOfNodes int) {

	honest.blockUpdates = honest.blockUpdates[:0]
}

func (honest *Honest) evaluateBlockQuality(block Block) bool{

	//TODO: This is just a simple equality check comparing the hashes. 
	myBlock := honest.bc.getBlock(block.Data.Iteration)

	// check equality
	if(string(block.PrevBlockHash[:]) == string(myBlock.PrevBlockHash[:]) && string(block.Hash[:]) == string(myBlock.Hash[:])) {
		return false
	}else{
		
		if (len(block.Data.Deltas) == 0){

			return false
		}
	
	}

	return true
	
}

func (honest *Honest) replaceBlock(block Block, iterationCount int){

	*honest.bc.blocks[iterationCount] = block

}

//Test the current global model. Determine training and test error to see if model has converged 

func testModel(weights []float64, node string) (float64, float64) {

	runtime.LockOSThread()

	argArray := python.PyList_New(len(weights))

	for i := 0; i < len(weights); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(weights[i]))
	}

	_gstate := python.PyGILState_Ensure()

	pyTrainResult := pyTrainFunc.CallFunction(argArray)

	trainErr := python.PyFloat_AsDouble(pyTrainResult)

	pyTestResult := pyTestFunc.CallFunction(argArray)

	python.PyGILState_Release(_gstate)

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

	filename := datasetName + strconv.Itoa(nodeID) + ".csv"
	file, err := os.Create(datasetPath + filename)
	check(err)
	nodeData.WriteCSV(bufio.NewWriter(file))

}

// reads data into dataframe

func getData(filePath string) dataframe.DataFrame {

	f, err := os.Open(filePath)
	handleErrorFatal("Dataset not found",err)
	df := dataframe.ReadCSV(bufio.NewReader(f))
	return df

}


// Error - checking

func check(e error) {

	if e != nil {
		panic(e)
	}

}

