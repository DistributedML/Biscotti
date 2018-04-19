package main

import(
	"fmt"
	"bufio"
	"os"
	"strconv"
	"encoding/binary"
	"math"
	// "encoding/csv"
	// "encoding/csv"
	// "kniren/gota"
	// "io"
	// "container/list"
	"github.com/kniren/gota/dataframe"
	"github.com/sbinet/go-python"
)

var(
	datasetPath string
	datasetName string

	numberOfNodes int
	epsilon 	float64
	numFeatures int
	minClients	int
	pulledGradient []float64
	deltas []float64
	// DataFrame dataframe.DataFrame
	// thisRecord []string
	pyLogModule       *python.PyObject
	pyLogInitFunc     *python.PyObject
	pyLogPrivFunc     *python.PyObject
	pyNumFeatures 	  *python.PyObject
)

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func main() {

	// Generating an SGD for a particular dataset using Go-Python
	
	// Take the dataset and divide it into appropriate number of csv files for go-python

	datasetPath = "../ML/data/"
	datasetName = "creditcard"
	numberOfNodes = 4
	epsilon = 1.0

	// Take the dataset and divide it into appropriate number of csv files for go-python

	data := getData(datasetPath+datasetName+".csv")	
	dividedData := divideData(data, numberOfNodes)

	for i := 0; i < numberOfNodes; i++ {
		createCSVs(dividedData, datasetName, i)
		pyInit(datasetName+strconv.Itoa(i))
		deltas, err := oneGradientStep(pulledGradient)	
		check(err)
		fmt.Println(len(deltas))

	}
	
	// pyInit("credit")


	
	

	// // y := data[0:100]
	// // fmt.Printf("Length of Node: %d", y.Nrow())

	


	// // Testing divided Data 

	// // fmt.Printf("Length of dividedData: %d", len(dividedData))

	// // for i := 0; i < numberOfNodes ; i++ {
	// // 	fmt.Printf("Length of Node: %d", dividedData[i].Nrow())
	// // 	fmt.Println(dividedData[i].Subset([]int{3}))
	// // }

	// for i := 0; i < numberOfNodes; i++ {
	// 	computeSGD(dividedData, nodeID)	
	// }

	






	

	
	// allData, _ := data.ReadAll()
	// dividedData = divideData(data,numberOfNodes)
	// computeSGD(dividedData, nodeID)

	// bc := NewBlockchain()


	// bData1 := BlockData{1, 3.0,[]Update{Update{1,3.0}} }
	// bData2 := BlockData{2, 9.0,[]Update{Update{1,3.0},Update{2,6.0}} }

	// bc.AddBlock(bData1)
	// bc.AddBlock(bData2)

	// for _, block := range bc.blocks {
	// 	fmt.Printf("Prev. hash: %x\n", block.PrevBlockHash)
	// 	fmt.Printf("Data: %s\n", block.data.String())
	// 	fmt.Printf("Hash: %x\n", block.Hash)
	// 	fmt.Println()
	// }
}

func getData(filePath string) dataframe.DataFrame{

	f, err:= os.Open(filePath)
	check(err)
	df := dataframe.ReadCSV(bufio.NewReader(f))
	return df
}

func check(e error) {
    if e != nil {
        panic(e)
    }
}


func divideData(data dataframe.DataFrame,numberOfNodes int) []dataframe.DataFrame{

	var dividedData []dataframe.DataFrame
	indexes := make([]int, 0)

	var stepsize int
	start:= 0
	end :=  0
	
	stepsize = data.Nrow()/numberOfNodes
	fmt.Printf("Stepsize:%d",stepsize)

	for i := 0; i < numberOfNodes; i++ {
		
		if(i==numberOfNodes-1){
			end = data.Nrow()
		}else{
			end = start+stepsize
		}


		for j := 0; j < (end - start); j++ {
			if(i==0){
				indexes = append(indexes, start+j)
			}else{

				if(j < len(indexes)){
					indexes[j] = start + j			
				}else{
					indexes = append(indexes, start+j)
				}
			}

		}
		dividedData = append(dividedData, data.Subset(indexes))
		start = start + stepsize
		
	}

	return dividedData

}

func createCSVs(dividedData []dataframe.DataFrame, datasetName string, nodeID int){	

	nodeData := dividedData[nodeID]
	filename :=  datasetName + strconv.Itoa(nodeID) + ".csv"
	file, err := os.Create(datasetPath + filename)
	check(err)
	nodeData.WriteCSV(bufio.NewWriter(file))
}



// func computeSGD{nodeData dataframe.DataFrame){

// 	pyInit(nodeData)

	

// } 

func pyInit(datasetName string) {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("../ML/code"))
	
	pyLogModule = python.PyImport_ImportModule("logistic_model")
	pyLogInitFunc = pyLogModule.GetAttrString("init")
	pyLogPrivFunc = pyLogModule.GetAttrString("privateFun")
	pyNumFeatures = pyLogInitFunc.CallFunction(python.PyString_FromString(datasetName), python.PyFloat_FromDouble(epsilon))
	

  	numFeatures = python.PyInt_AsLong(pyNumFeatures)  	
  	minClients = 5
  	pulledGradient = make([]float64, numFeatures)
  	deltas = make([]float64, numFeatures)

  	fmt.Printf("Sucessfully pulled dataset. Features: %d\n", numFeatures)

  	
  	
}

func oneGradientStep(globalW []float64) ([]float64, error) {
	
	argArray := python.PyList_New(len(globalW))

	for i := 0; i < len(globalW); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(globalW[i]))
	}

	// Either use full GD or SGD here
	result := pyLogPrivFunc.CallFunction(python.PyInt_FromLong(1), argArray,
		python.PyInt_FromLong(10))
	
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