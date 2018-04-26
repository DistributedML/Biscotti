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
	"gonum.org/v1/gonum/mat"
	"github.com/DistributedClocks/GoVector/govec"

)

const (
	basePort	int 	= 8000
	myIP		string 	= "127.0.0.1"
)

var(
	datasetPath 	= "../ML/data/"
	datasetName 	string
	batch_size		int

	myPort			string
	portsToConnect  []string
	clusterPorts    []string
	client 			Honest

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

	logger          *govec.GoLog
)

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func main() {

	// some declarations

	nodeNum, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Println("First argument should be index of node")
		return
	}

	nodeTotal, err := strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Println("Second argument should be the total number of nodes")
		return
	}
	
	datasetName := os.Args[3]
	if err != nil {
		fmt.Println("Third argument should be dataset name")
		return
	}

	fmt.Println(nodeNum)
	fmt.Println(nodeTotal)
	fmt.Println(datasetName)


	
	datasetName = "creditcard"
	numberOfNodes = 4
	epsilon = 1.0
	batch_size = 10

	// Take the dataset and divide it into appropriate number of csv files for go-python
	// Once divided, compute SGD using go-python

	logger = govec.InitGoVector(os.Args[1], os.Args[1])
	myPort = strconv.Itoa(nodeNum + basePort)
	for i := 0; i < nodeTotal; i++ {
		if strconv.Itoa(basePort+i) == myPort {
			continue
		}
		clusterPorts = append(clusterPorts, strconv.Itoa(basePort+i))
	}
	client = Honest{id: nodeNum}


	data := getData(datasetPath+datasetName+".csv")
  	deltas = make([]float64, data.Ncol())
  	pulledGradient = make([]float64, data.Ncol())
	pulledGradientM := mat.NewDense(1, data.Ncol(), pulledGradient)
	deltaM := mat.NewDense(1,data.Ncol(), deltas)

	bc := NewBlockchain(data.Ncol())

	dividedData := divideData(data, numberOfNodes)	

	for i := 0; i < numberOfNodes; i++ {

		createCSVs(dividedData, datasetName, i)
		pyInit(datasetName+strconv.Itoa(i))
		deltas, err := oneGradientStep(pulledGradient)
		deltaM = mat.NewDense(1, data.Ncol(), deltas)		 		
		pulledGradientM.Add(pulledGradientM,deltaM)
		mat.Row(pulledGradient,0,pulledGradientM)
		bData := BlockData{i, pulledGradient,[]Update{Update{deltas}} } // globalW do this
		check(err)
		bc.AddBlock(bData)
		fmt.Println(len(deltas))	
	}

	for _, block := range bc.blocks {
		fmt.Printf("Prev. hash: %x\n", block.PrevBlockHash)
		fmt.Printf("Data: %s\n", block.data.String())
		fmt.Printf("Hash: %x\n", block.Hash)
		fmt.Println()
	}

	// bData2 := BlockData{2, 9.0,[]Update{Update{1,3.0},Update{2,6.0}} }
	// pushing it as a block update

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