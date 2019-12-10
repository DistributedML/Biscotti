package main

import (
	// "math/rand"
	"encoding/binary"
	"github.com/sbinet/go-python"
	"gonum.org/v1/gonum/mat"
	"math"
	"strconv"
	"errors"
	"runtime"
	"math/rand"
	"time"
)

var (
	
	pyLogModule   *python.PyObject
	pyTorchModule 		*python.PyObject

	pyInitFunc *python.PyObject
	pyPrivFunc *python.PyObject
	pyNumFeatures *python.PyObject
	pyTestModule  *python.PyObject
	pyTestFunc    *python.PyObject
	pyTrainFunc   *python.PyObject
	pyRoniModule  *python.PyObject
	pyRoniFunc    *python.PyObject
	pyNoiseFunc	  *python.PyObject
	pyAttackFunc    *python.PyObject

	useTorch	   bool

	//Errors
	 blockExistsError = errors.New("Forbidden overwrite of block foiled")
)

const (

	STAKE_UNIT		= 5

	batch_size      = 10
	datasetPath     = "../ML/data/"
	codePath        = "../ML/code"
	torchPath       = "../ML/Pytorch"
	convThreshold   = 0.00

	// Crypto constants
	commitKeyPath = "commitKey.json"
	pKeyG1Path = "pKeyG1.json"	

)

type Honest struct {
	
	id           		int
	dataset 	 		string
	ncol 	     		int
	update       		Update
	blockUpdates 		[]Update
	globalModel	 		[]float64
	iteration 			int
}

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

// Load data and initialize chain

func (honest *Honest) initializeData(datasetName string, numberOfNodes int, epsilon float64, isPoisoning bool) {

	if datasetName == "creditcard" {
		
		useTorch = false
	
		if isPoisoning {
			outLog.Println("Get the bad credit data.")
			honest.ncol = pyInit("creditbad", "creditbad", epsilon)	
		} else {
			honest.ncol = pyInit(datasetName, datasetName + strconv.Itoa(honest.id), epsilon)
		}	

	} else {
	
		useTorch = true
	
		if isPoisoning {
			outLog.Println("Get the bad data.")
			honest.ncol = pyInit("mnist", "mnist_bad", epsilon)	
		} else {
			honest.ncol = pyInit(datasetName, datasetName + strconv.Itoa(honest.id), epsilon)
		}

	}
	
	honest.dataset = datasetName
	honest.globalModel = make([]float64, honest.ncol)

}

// check for Convergence by calling TestModel that invokes puython to compute train and test error 
func (honest *Honest) checkConvergence(iterationCount int) bool {

	trainError := testModel(honest.globalModel)

	if honest.dataset == "creditcard" {
		trainError := testModel(honest.globalModel)
		outLog.Printf(strconv.Itoa(honest.id)+":Train Error is %.5f in Iteration %d",
			trainError, iterationCount)
	} else {
		attackRate := testAttackRate(honest.globalModel)
		outLog.Printf(strconv.Itoa(honest.id)+":Train Error is %.5f in Iteration %d",
			trainError, iterationCount)
		outLog.Printf(strconv.Itoa(honest.id)+":Attack Rate is %.5f in Iteration %d",
			attackRate, iterationCount)
	}

	if trainError < convThreshold {
		return true
	}

	return false
}

// cSample updates for the poisoning case
func (honest *Honest) sampleUpdates(numUpdates int) {

	// s := rand.New(rand.NewSource(time.Now().Unix()))
	// r := rand.New(s)
	rand.Seed(time.Now().UnixNano())
	selectedUpdates := make([]Update, numUpdates)
	selectedIdxs := make([]int, numUpdates)
	// perm := r.Perm(len(honest.blockUpdates))


	for i := 0; i < numUpdates; i++ {
		randIndex := rand.Intn(len(honest.blockUpdates))
		selectedIdxs[i] = randIndex
		selectedUpdates[i] = honest.blockUpdates[randIndex]
	}
	
	// for i, randIndex := range perm {
	// 	selectedUpdates[i] = honest.blockUpdates[randIndex]

	// 	if i == (numUpdates-1) {
	// 		break
	// 	}
	
	// }	

	honest.blockUpdates = selectedUpdates

	outLog.Printf("Number of updates sampled:%s", len(honest.blockUpdates))
	// outLog.Printf("Indexes selected:%s", perm[:numUpdates])
	outLog.Printf("Indexes selected:%s", selectedIdxs)



}

// calculates update by calling oneGradientStep function that invokes python and passing latest global model from the chain to it.
func (honest *Honest) computeUpdate(iterationCount int) {
	
	prevModel := honest.globalModel
	
	deltas, err := oneGradientStep(prevModel) // TODO: Create commitment here

	check(err)

	/* outLog.Printf("Global Model: %v", prevModel)
	outLog.Printf("Deltas: %v", deltas)*/
	
	honest.update = Update {
		SourceID: honest.id,
		Iteration: iterationCount, 
		Delta: deltas }

}

// Initialize the python stuff using go-python

func pyInit(datasetName string, dataFile string, epsilon float64) int {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	
	outLog.Printf("Importing modules...")

	// We currently support creditcard for logreg, and mnist/lfw for pytorch
	if useTorch {
	
		// Get all PyTorch related
		python.PyList_Insert(sysPath, 0, python.PyString_FromString(torchPath))
	    
	    pyTorchModule = python.PyImport_ImportModule("client_obj")

		pyInitFunc = pyTorchModule.GetAttrString("init")
		pyPrivFunc = pyTorchModule.GetAttrString("privateFun")
		pyTrainFunc = pyTorchModule.GetAttrString("getTestErr")
		pyTestFunc = pyTorchModule.GetAttrString("getTestErr")
		pyRoniFunc = pyTorchModule.GetAttrString("roni")
		pyNoiseFunc = pyTorchModule.GetAttrString("getNoise")
		pyAttackFunc = pyTorchModule.GetAttrString("get17AttackRate")

	} else {
		
		// Get all others
		python.PyList_Insert(sysPath, 0, python.PyString_FromString(codePath))

		pyLogModule = python.PyImport_ImportModule("logistic_model")
		pyTestModule = python.PyImport_ImportModule("logistic_model_test")
		pyRoniModule = python.PyImport_ImportModule("logistic_validator")

		pyInitFunc = pyLogModule.GetAttrString("init")
		pyPrivFunc = pyLogModule.GetAttrString("privateFun")
		pyNoiseFunc = pyLogModule.GetAttrString("getNoise")
		pyTrainFunc = pyTestModule.GetAttrString("train_error")
		pyTestFunc = pyTestModule.GetAttrString("test_error")
		pyRoniFunc = pyRoniModule.GetAttrString("roni")

	}
	
	// If epsilon is 0, this tells python not to pre-sample noise, 
	// which saves a lot of time and memory
	pyNumFeatures = pyInitFunc.CallFunction(python.PyString_FromString(datasetName), 
			python.PyString_FromString(dataFile), python.PyFloat_FromDouble(epsilon),
			python.PyInt_FromLong(batch_size))

	numFeatures := python.PyInt_AsLong(pyNumFeatures)

	outLog.Printf("Sucessfully pulled dataset. Features: %d\n", numFeatures)

	return numFeatures

}

func testAttackRate(weights []float64) float64 {

	runtime.LockOSThread()

	_gstate := python.PyGILState_Ensure()

	argArray := python.PyList_New(len(weights))

	for i := 0; i < len(weights); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(weights[i]))
	}

	var attackRate float64
	pyTrainResult := pyAttackFunc.CallFunction(argArray)
	attackRate = python.PyFloat_AsDouble(pyTrainResult)

	python.PyGILState_Release(_gstate)	

	return attackRate

}
// calculate the next update using the latest global model on the chain invoking python

func oneGradientStep(globalW []float64) ([]float64, error) {

	runtime.LockOSThread()

	_gstate := python.PyGILState_Ensure()

	argArray := python.PyList_New(len(globalW))

	for i := 0; i < len(globalW); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(globalW[i]))
	}
	
	var result *python.PyObject
	result = pyPrivFunc.CallFunction(argArray)

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

func (honest *Honest) createNewModel(iterationCount int) (BlockData, error) {

	pulledGradient := make([]float64, honest.ncol)
	pulledGradient = honest.globalModel
	updatedGradient := make([]float64, honest.ncol)
	deltaM := mat.NewDense(1, honest.ncol, make([]float64, honest.ncol))
	pulledGradientM := mat.NewDense(1, honest.ncol, pulledGradient)
	// avgFactor := 1.0/float64(len(honest.blockUpdates))

	// Update Aggregation
	for _, update := range honest.blockUpdates {
		deltaM = mat.NewDense(1, honest.ncol, update.Delta)
		pulledGradientM.Add(pulledGradientM, deltaM)	
	}

	// pulledGradientM.Scale(avgFactor, pulledGradientM)

	mat.Row(updatedGradient, 0, pulledGradientM)

	updatesGathered := make([]Update, len(honest.blockUpdates))
	copy(updatesGathered, honest.blockUpdates)

	bData := BlockData{iterationCount, updatedGradient, updatesGathered}

	return bData, nil

}

// Empty the updates recorded at the start of each iteration

func (honest *Honest) flushUpdates() {

	honest.blockUpdates = honest.blockUpdates[:0]
}


//Test the current global model. Determine training and test error to see if model has converged 
func testModel(weights []float64) float64 {

	runtime.LockOSThread()

	_gstate := python.PyGILState_Ensure()

	argArray := python.PyList_New(len(weights))

	for i := 0; i < len(weights); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(weights[i]))
	}

	var trainErr float64
	pyTrainResult := pyTrainFunc.CallFunction(argArray)
	trainErr = python.PyFloat_AsDouble(pyTrainResult)

	python.PyGILState_Release(_gstate)	

	return trainErr

}

// Error - checking
func check(e error) {

	if e != nil {
		panic(e)
	}

}
