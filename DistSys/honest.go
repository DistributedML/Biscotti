package main

import (
	// "math/rand"
	"fmt"
	"encoding/binary"
	"math"
	"strconv"
	"gonum.org/v1/gonum/mat"
	"github.com/kniren/gota/dataframe"
	"github.com/sbinet/go-python"
)

var(

	pyLogModule       *python.PyObject
	pyLogInitFunc     *python.PyObject
	pyLogPrivFunc     *python.PyObject
	pyNumFeatures 	  *python.PyObject
	pyTestModule  	  *python.PyObject
	pyTestFunc    	  *python.PyObject
	pyTrainFunc   	  *python.PyObject

)

const (
	samples         = 10 // L
	sampleDeviation = 0.1
	batch_size 		= 10
	epsilon			= 1.0
	datasetPath 	=  "../ML/data/"
	codePath		=  "../ML/code"
	convThreshold	= 0.05
)

// Honest Client
type Honest struct {
	id        		int
	data 			dataframe.DataFrame 
	update    		Update
	blockUpdates	[]Update 
	bc 				*Blockchain
}

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func (honest *Honest) initializeData(datasetName string, numberOfNodes int) {

	fullData := getData(datasetPath+datasetName+".csv")
	honest.data = divideData(fullData, numberOfNodes)[honest.id]
	createCSVs(honest.data, datasetName, honest.id)
	honest.bc = NewBlockchain(honest.data.Ncol())	
	pyInit(datasetName+strconv.Itoa(honest.id))

}

func (honest *Honest) checkConvergence() bool {

	trainError, _ := testModel(honest.bc.getLatestGradient(), "global")
	fmt.Printf("Train Error: %d\n",trainError)
	if (trainError<convThreshold){
		return true
	}
	return false
}

func (honest *Honest) computeUpdate(iterationCount int,datasetName string){
	prevGradient := honest.bc.getLatestGradient()
	deltas, err := oneGradientStep(prevGradient)
	check(err)
	honest.update= Update{Iteration:iterationCount, Delta:deltas}	
	// fmt.Println(honest.update)
}

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
	numFeatures = python.PyInt_AsLong(pyNumFeatures)  	
  	minClients = 5  	
  	// deltas = make([]float64, numFeatures)

  	fmt.Printf("Sucessfully pulled dataset. Features: %d\n", numFeatures)
  	
}

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

func (honest *Honest) addBlockUpdate(update Update) int{	
	
	honest.blockUpdates = append(honest.blockUpdates,update)
	// fmt.Println(honest.blockUpdates)
	return len(honest.blockUpdates)
}

func (honest *Honest) createBlock(iterationCount int) Block{

	pulledGradient = make([]float64, honest.data.Ncol())
	pulledGradient = honest.bc.getLatestGradient()
	updatedGradient := make([]float64, honest.data.Ncol()) 
	deltaM := mat.NewDense(1, honest.data.Ncol() , make([]float64, honest.data.Ncol()))
	pulledGradientM := mat.NewDense(1, honest.data.Ncol(), pulledGradient)

	// Update Aggregation
	for _, update := range honest.blockUpdates {			
		deltaM = mat.NewDense(1, honest.data.Ncol(), update.Delta)
		pulledGradientM.Add(pulledGradientM,deltaM)
	}
	mat.Row(updatedGradient,0,pulledGradientM)

	// TODO: Insert RONI

	// Block Creation
	bData := BlockData{iterationCount, updatedGradient, honest.blockUpdates }
	honest.bc.AddBlock(bData) // not sure whether this should be here. For now, seems okay

	var newBlock Block
	newBlock = *(honest.bc.blocks[len(honest.bc.blocks) - 1])

	return newBlock


	// Just testing block creation in the chain. next TODO: Send this to all nodes that connected


}

func (honest *Honest) flushUpdates(numberOfNodes int) {

	// if(len(honest.blockUpdates) > numberOfNodes){		
		// honest.blockUpdates = honest.blockUpdates[numberOfNodes:len(honest.blockUpdates)]		
	// }else{
		honest.blockUpdates = honest.blockUpdates[:0]		
	// }
}

func testModel(weights []float64, node string) (float64, float64) {

	argArray := python.PyList_New(len(weights))

	for i := 0; i < len(weights); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(weights[i]))
	}

    pyTrainResult := pyTrainFunc.CallFunction(argArray)
    trainErr := python.PyFloat_AsDouble(pyTrainResult)

	pyTestResult := pyTestFunc.CallFunction(argArray)
	testErr := python.PyFloat_AsDouble(pyTestResult)
	
	return trainErr, testErr

}





func (honest *Honest) aggregateUpdates(updates []Update) float64 {

	// for _, update := range len(updates) {


	// }

	// sum := 0.0
	// for _, update := range updates {
	// 	sum += update.Value
	// }
	// return sum / float64(len(updates))

	return 1.00
}

