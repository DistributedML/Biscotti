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
	pyRoniModule  *python.PyObject
	pyRoniFunc    *python.PyObject

	pyTorchModule 		*python.PyObject
	pyTorchInitFunc     *python.PyObject
	pyTorchPrivFunc     *python.PyObject
	pyTorchErrFunc      *python.PyObject
	pyTorchRoniFunc 	*python.PyObject

	useTorch	   bool

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
	torchPath       = "../ML/Pytorch"
	convThreshold   = 0.05
)

type Honest struct {
	id           int
	dataset 	 string
	ncol 	     int
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

	if datasetName == "creditcard" {
		useTorch = false
	} else {
		useTorch = true
	}

	honest.ncol = pyInit(datasetName, datasetName + strconv.Itoa(honest.id))
	honest.dataset = datasetName
	honest.bc = NewBlockchain(honest.ncol)

}

// check for Convergence by calling TestModel that invokes puython to compute train and test error 

func (honest *Honest) checkConvergence() bool {

	// outLog.Println(honest.bc.getLatestGradient()[4000:4010])
	trainError, _ := testModel(honest.bc.getLatestGradient())

	outLog.Printf(strconv.Itoa(client.id)+":Train Error is %.5f in Iteration %d", trainError, honest.bc.Blocks[len(honest.bc.Blocks)-1].Data.Iteration)

	if trainError < convThreshold {
		return true
	}

	return false
}

// calculates update by calling oneGradientStep function that invokes python and passing latest global model from the chain to it.

func (honest *Honest) computeUpdate(iterationCount int, datasetName string) {
	prevModel := honest.bc.getLatestGradient()
	deltas, err := oneGradientStep(prevModel)
	check(err)
	honest.update = Update{Iteration: iterationCount, Delta: deltas,
		Accepted: true}
}

// Initialize the python stuff using go-python

func pyInit(datasetName string, dataFile string) int {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	
	outLog.Printf(strconv.Itoa(client.id)+"Importing modules...")

	// Get all PyTorch related
	python.PyList_Insert(sysPath, 0, python.PyString_FromString(torchPath))
    pyTorchModule = python.PyImport_ImportModule("client_obj")

	pyTorchInitFunc = pyTorchModule.GetAttrString("init")
	pyTorchPrivFunc = pyTorchModule.GetAttrString("privateFun")
	pyTorchErrFunc = pyTorchModule.GetAttrString("getTestErr")
	pyTorchRoniFunc = pyTorchModule.GetAttrString("roni")

	// Get all others
	python.PyList_Insert(sysPath, 0, python.PyString_FromString(codePath))

	pyLogModule = python.PyImport_ImportModule("logistic_model")
	pyTestModule = python.PyImport_ImportModule("logistic_model_test")
	pyRoniModule = python.PyImport_ImportModule("logistic_validator")

	pyLogInitFunc = pyLogModule.GetAttrString("init")
	pyLogPrivFunc = pyLogModule.GetAttrString("privateFun")
	pyTrainFunc = pyTestModule.GetAttrString("train_error")
	pyTestFunc = pyTestModule.GetAttrString("test_error")
	pyRoniFunc = pyRoniModule.GetAttrString("roni")

	// TODO: clean up
	// We currently support creditcard for logreg, and mnist/lfw for pytorch
	if useTorch {
		pyNumFeatures = pyTorchInitFunc.CallFunction(python.PyString_FromString(datasetName), 
			python.PyString_FromString(dataFile))
	} else {
		pyNumFeatures = pyLogInitFunc.CallFunction(python.PyString_FromString(dataFile), 
			python.PyFloat_FromDouble(epsilon))
	}
	
	numFeatures := python.PyInt_AsLong(pyNumFeatures)

	outLog.Printf(strconv.Itoa(client.id)+"Sucessfully pulled dataset. Features: %d\n", numFeatures)

	return numFeatures

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
	var result *python.PyObject
	if useTorch {
		result = pyTorchPrivFunc.CallFunction(argArray)
	} else { 
		result = pyLogPrivFunc.CallFunction(python.PyInt_FromLong(1), 
			argArray, python.PyInt_FromLong(batch_size))
	}

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

	pulledGradient := make([]float64, honest.ncol)
	pulledGradient = honest.bc.getLatestGradient()
	updatedGradient := make([]float64, honest.ncol)
	deltaM := mat.NewDense(1, honest.ncol, make([]float64, honest.ncol))
	pulledGradientM := mat.NewDense(1, honest.ncol, pulledGradient)

	// Update Aggregation
	for _, update := range honest.blockUpdates {
		if update.Accepted {
			deltaM = mat.NewDense(1, honest.ncol, update.Delta)
			pulledGradientM.Add(pulledGradientM, deltaM)	
		} else {
			outLog.Printf("Skipping an update")
		}
		
	}
	mat.Row(updatedGradient, 0, pulledGradientM)

	updatesGathered := make([]Update, len(honest.blockUpdates))
	copy(updatesGathered, honest.blockUpdates)

	bData := BlockData{iterationCount, updatedGradient, updatesGathered}
	honest.bc.AddBlock(bData) 

	newBlock := honest.bc.Blocks[len(honest.bc.Blocks)-1]

	return newBlock,nil


}

// function to check if you have a block with the same iteration
func (honest *Honest) hasBlock(iterationCount int) bool {
	
	if (honest.bc.getBlock(iterationCount) != nil) {
		return true;
	} else {
		return false;
	}

}

func (honest *Honest) addBlock(newBlock Block) error {

	// if already exists don't create/replace it
	outLog.Printf("Trying to append block with iteration:%d", newBlock.Data.Iteration)
	// outLog.Println(honest.bc.getLatestGradient()[4000:4010])

	if(honest.bc.getBlock(newBlock.Data.Iteration) != nil){
		
		better := honest.evaluateBlockQuality(newBlock)
		if(!better){
			outLog.Printf("Append foiled")
			return blockExistsError
		}else{
			outLog.Printf("Replace successful")
			honest.replaceBlock(newBlock, newBlock.Data.Iteration) // this thing doesn't need the second argument I think
			return nil
		}
	
	}else{
		
		outLog.Printf("Append successful")
		honest.bc.AddBlockMsg(newBlock)
		// outLog.Println(honest.bc.getLatestGradient()[4000:4010])
		return nil

	}

}

// Empty the updates recorded at the start of each iteration

func (honest *Honest) flushUpdates() {

	honest.blockUpdates = honest.blockUpdates[:0]
}


/*
	Runs RONI through python on the proposed update
*/
func (honest *Honest) verifyUpdate(update Update) float64 {

	runtime.LockOSThread()

	_gstate := python.PyGILState_Ensure()

	deltas := update.Delta
	truthModel := honest.bc.getLatestGradient()

	truthArray := python.PyList_New(len(truthModel))
	updateArray := python.PyList_New(len(truthModel))

	// Convert both into PyArrays
	for i := 0; i < len(truthModel); i++ {
		python.PyList_SetItem(truthArray, i, python.PyFloat_FromDouble(truthModel[i]))
		python.PyList_SetItem(updateArray, i, python.PyFloat_FromDouble(deltas[i]))
	}

	var score float64
	if useTorch {
		pyRoni := pyTorchRoniFunc.CallFunction(truthArray, updateArray)
		score = python.PyFloat_AsDouble(pyRoni)
	} else {
		pyRoni := pyRoniFunc.CallFunction(truthArray, updateArray)
		score = python.PyFloat_AsDouble(pyRoni)		
	}
	
	python.PyGILState_Release(_gstate)	

	return score

}

func (honest *Honest) evaluateBlockQuality(block Block) bool {

	//TODO: This is just a simple equality check comparing the hashes. 
	myBlock := honest.bc.getBlock(block.Data.Iteration)
	previousBlock := honest.bc.getBlock(block.Data.Iteration-1)

	// check equality
	if(string(block.PrevBlockHash[:]) != string(previousBlock.Hash[:])) {		
		outLog.Printf("Inconsistent hashes. ThisHash:" + string(block.PrevBlockHash[:]) +".Previous Hash:" + string(previousBlock.Hash[:]) )
		return false
	} else if (len(block.Data.Deltas) == 0 || len(myBlock.Data.Deltas) != 0) {
		return false
	}
	
	return true
	
}

func (honest *Honest) replaceBlock(block Block, iterationCount int){

	*honest.bc.Blocks[iterationCount+1] = block

}

//Test the current global model. Determine training and test error to see if model has converged 
func testModel(weights []float64) (float64, float64) {

	runtime.LockOSThread()

	_gstate := python.PyGILState_Ensure()

	argArray := python.PyList_New(len(weights))

	for i := 0; i < len(weights); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(weights[i]))
	}

	var trainErr, testErr float64
	if useTorch {
		pyTrainResult := pyTorchErrFunc.CallFunction(argArray)
		trainErr = python.PyFloat_AsDouble(pyTrainResult)
		testErr = trainErr

	} else {
		pyTrainResult := pyTrainFunc.CallFunction(argArray)
		trainErr = python.PyFloat_AsDouble(pyTrainResult)

		pyTestResult := pyTestFunc.CallFunction(argArray)
		testErr = python.PyFloat_AsDouble(pyTestResult)
	}

	python.PyGILState_Release(_gstate)	

	return trainErr, testErr

}

// replace the current chain with the one input and return the latest iteration count
func (honest *Honest) replaceChain(chain Blockchain) int {
	
	*honest.bc = chain
	outLog.Printf("Received chain length:%d",  len(chain.Blocks))
	outLog.Printf("Appended chain length:%d",  len(honest.bc.Blocks))	
	return chain.Blocks[len(chain.Blocks) - 1].Data.Iteration
}

// DEPRECATED: Divide the dataset equally among the number of nodes
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


// DEPRECATED: creates a CSV for your part of the data
func createCSVs(nodeData dataframe.DataFrame, datasetName string, nodeID int) {

	filename := datasetName + strconv.Itoa(nodeID) + ".csv"
	file, err := os.Create(datasetPath + filename)
	check(err)
	nodeData.WriteCSV(bufio.NewWriter(file))

}

// DEPRECATED: reads data into dataframe
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

