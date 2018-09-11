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
	"github.com/dedis/kyber"
	"github.com/dedis/kyber/pairing/bn256"
	"encoding/json"
	// "fmt"
	"sort"

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
	bc           		*Blockchain
	Keys 		 		EncryptionKeys
	secretList	        map[int]MinerPart	
	aggregatedSecrets   []MinerPart
}

type EncryptionKeys struct{

	CommitmentKey PublicKey
	PubKey  	  kyber.Point 
	PubKeyMap     map[int]PublicKey
	Skey 		  kyber.Scalar
} 

type PkeyG1 struct{

	Id 		int
	Pkey 	[]byte
	Skey 	[]byte

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

// Load all the public and private keys

func (honest *Honest) bootstrapKeys() {

	honest.Keys.CommitmentKey = extractCommitmentKey(honest.ncol)
	honest.Keys.PubKeyMap, honest.Keys.Skey, honest.Keys.PubKey = extractKeys(honest.id)
	// fmt.Println(honest.Keys.PubKeyMap)
	// fmt.Println(honest.Keys.Skey)
	// fmt.Println(honest.Keys.PubKey)
	// fmt.Println(honest.Keys.CommitmentKey)

}

// check for Convergence by calling TestModel that invokes puython to compute train and test error 
func (honest *Honest) checkConvergence() bool {

	trainError := testModel(honest.bc.getLatestGradient())

	outLog.Printf(strconv.Itoa(client.id)+":Train Error is %.5f in Iteration %d", trainError, honest.bc.Blocks[len(honest.bc.Blocks)-1].Data.Iteration)

	if trainError < convThreshold {
		return true
	}

	return false
}

// calculates update by calling oneGradientStep function that invokes python and passing latest global model from the chain to it.
func (honest *Honest) computeUpdate(iterationCount int) {
	prevModel := honest.bc.getLatestGradient()
	deltas, err := oneGradientStep(prevModel) // TODO: Create commitment here
	outLog.Printf("This update float:%s", deltas)
	check(err)	
	deltasInt := updateFloatToInt(deltas, PRECISION)
	outLog.Printf("This update:%s", deltasInt)
	updateCommitment := createCommitment(deltasInt, client.Keys.CommitmentKey.PKG1)
	byteCommitment, err := updateCommitment.MarshalBinary()
	check(err)
	honest.update = Update{Iteration: iterationCount, 
		Commitment: byteCommitment,
		Delta: deltas, 
		NoisedDelta: deltas, 
		Noise: deltas,
		Accepted: true}

}

// Initialize the python stuff using go-python

func pyInit(datasetName string, dataFile string) int {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	
	outLog.Printf(strconv.Itoa(client.id)+"Importing modules...")

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
	
	pyNumFeatures = pyInitFunc.CallFunction(python.PyString_FromString(datasetName), 
			python.PyString_FromString(dataFile), python.PyFloat_FromDouble(epsilon))

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
	result = pyPrivFunc.CallFunction(argArray, python.PyInt_FromLong(batch_size))

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

// add an update to the record of updates received for the current iteration

func (honest *Honest) addSecretShare(share MinerPart) int {

	honest.secretList[share.NodeID] = share
	return len(honest.secretList)
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

// creates a block from all the updates recorded.

func (honest *Honest) createBlockSecAgg(iteration int, nodeList []int) (*Block,error) {

	// Has block already been appended from advertisements by other client?
	if(honest.bc.getBlock(iterationCount) != nil){
		return nil, blockExistsError
	}

	pulledGradient := make([]float64, honest.ncol)
	pulledGradient = honest.bc.getLatestGradient()
	updatedGradient := make([]float64, honest.ncol)
	deltaM := mat.NewDense(1, honest.ncol, make([]float64, honest.ncol))
	pulledGradientM := mat.NewDense(1, honest.ncol, pulledGradient)

		


	// Recover Secret Secure Aggregation
	
	if (len(nodeList) > 0){

		aggregateUpdate := honest.recoverAggregateUpdates()	
		deltaM = mat.NewDense(1, honest.ncol, aggregateUpdate)
		pulledGradientM.Add(pulledGradientM, deltaM)

	}
	

	// Update Aggregation
	for _, nodeIndex := range nodeList {
		
		byteCommitment, _ := honest.secretList[nodeIndex].CommitmentUpdate.MarshalBinary()
		thisNodeUpdate := Update{Iteration:iteration, Commitment: byteCommitment, Accepted:true}
		honest.blockUpdates = append(honest.blockUpdates, thisNodeUpdate)

		outLog.Printf("Update:%s", thisNodeUpdate)
		outLog.Printf("List of Updates:%s", honest.blockUpdates)
	
	}



	mat.Row(updatedGradient, 0, pulledGradientM)

	updatesGathered := make([]Update, len(honest.blockUpdates))
	copy(updatesGathered, honest.blockUpdates)

	bData := BlockData{iteration, updatedGradient, updatesGathered}
	honest.bc.AddBlock(bData) 

	newBlock := honest.bc.Blocks[len(honest.bc.Blocks)-1]

	return newBlock,nil

}

func (honest *Honest) recoverAggregateUpdates() []float64{

	 myIndex := 0

	 for index, subPolyPart := range honest.aggregatedSecrets[myIndex].PolyMap{

		 listOfShares := make([]Share,0)

		 for i := 0; i < len(honest.aggregatedSecrets); i++ {
		 	
		 	for _, share := range honest.aggregatedSecrets[i].PolyMap[index].Secrets{

		 		listOfShares = append(listOfShares, share)	
		 	}	 	
		 	
		 }

		 outLog.Printf("List of shares for index %d: %s", index, listOfShares)

		 subPolyPart.Polynomial = recoverSecret(listOfShares, POLY_SIZE-1)
		 honest.aggregatedSecrets[myIndex].PolyMap[index] = subPolyPart
		 outLog.Printf("Polynomial: %s" , subPolyPart.Polynomial)
		 outLog.Printf("Polynomial2: %s" , honest.aggregatedSecrets[myIndex].PolyMap[index].Polynomial)	 

	 }

	 reconstructedUpdate := make([]int64,0)
	 indexes := make([]int, 0)

	 for k, _ := range honest.aggregatedSecrets[myIndex].PolyMap {
	    indexes = append(indexes, k)
	}
	
	sort.Ints(indexes)	

	for _, index := range indexes{

		subPolyPart := honest.aggregatedSecrets[myIndex].PolyMap[index]

		outLog.Printf("Index:%d", index)
		outLog.Printf("Length:%d", len(reconstructedUpdate))
		outLog.Printf("Polynomial:%s", subPolyPart.Polynomial)


		for i := len(reconstructedUpdate); i < index; i++ {
 			
 			reconstructedUpdate = append(reconstructedUpdate, subPolyPart.Polynomial[i%POLY_SIZE])
 		
 		}

	}	 
	 
    // fmt.Println(reconstructedUpdate)

	aggregatedVectorFloat := updateIntToFloat(reconstructedUpdate, PRECISION)

	return aggregatedVectorFloat



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
		return nil

	}

}

// Empty the updates recorded at the start of each iteration

func (honest *Honest) flushUpdates() {

	honest.blockUpdates = honest.blockUpdates[:0]
}

// Empty the updates recorded at the start of each iteration

func (honest *Honest) flushSecrets() {

	// IFFY. How to empty a map, I don't know
	honest.secretList = make(map[int]MinerPart)
	honest.blockUpdates = honest.blockUpdates[:0]
	honest.aggregatedSecrets = honest.aggregatedSecrets[:0]


}


/*
	Return noise to a requesting client 
*/
func (honest *Honest) requestNoise(iterationCount int) ([]float64, error) {

	runtime.LockOSThread()

	_gstate := python.PyGILState_Ensure()

	// Either use full GD or SGD here
	var result *python.PyObject
	result = pyNoiseFunc.CallFunction(python.PyInt_FromLong(iterationCount))

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


/*
	Runs RONI through python on the proposed update
*/
func (honest *Honest) verifyUpdate(update Update) float64 {

	runtime.LockOSThread()

	_gstate := python.PyGILState_Ensure()

	deltas := update.NoisedDelta
	truthModel := honest.bc.getLatestGradient()

	truthArray := python.PyList_New(len(truthModel))
	updateArray := python.PyList_New(len(truthModel))

	// Convert both into PyArrays
	for i := 0; i < len(truthModel); i++ {
		python.PyList_SetItem(truthArray, i, python.PyFloat_FromDouble(truthModel[i]))
		python.PyList_SetItem(updateArray, i, python.PyFloat_FromDouble(deltas[i]))
	}

	var score float64
	pyRoni := pyRoniFunc.CallFunction(truthArray, updateArray)
	score = python.PyFloat_AsDouble(pyRoni)		
	
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

func extractKeys(nodeNum int) (map[int]PublicKey, kyber.Scalar, kyber.Point){
	
	pubKeyMap := make(map[int]PublicKey)

	suite := bn256.NewSuite()

	mySkey := suite.G1().Scalar().One()

	myPubKey := suite.G1().Point()

	thisPoint := suite.G1().Point().Null()

	pKeyG1File, err := os.Open(pKeyG1Path)
	check(err)

	defer pKeyG1File.Close()

	scanner := bufio.NewScanner(pKeyG1File)

	for scanner.Scan() {

		thisKeyBytes := scanner.Bytes()
		
		thisKey := PkeyG1{}

		json.Unmarshal(thisKeyBytes, &thisKey)		

		err = thisPoint.UnmarshalBinary(thisKey.Pkey)

		check(err)

		thisPubKey := PublicKey{}

		thisPubKey.SetG1Key(thisPoint.Clone())

		pubKeyMap[thisKey.Id] = thisPubKey 	

		// Write Set Key function for this 

		// pubKeyMap[thisKey.Id].PKG1[0] = thisPoint

		// fmt.Println(thisPoint)
		

		if(thisKey.Id == nodeNum){

			mySkey.UnmarshalBinary(thisKey.Skey)
			myPubKey = thisPoint.Clone()			
		
		}


	}

	return pubKeyMap, mySkey, myPubKey	

}

func extractCommitmentKey(dimensions int) PublicKey {	

	suite := bn256.NewSuite()

	commitKey := PublicKey{PKG1:make([]kyber.Point, dimensions), PKG2:make([]kyber.Point, dimensions)}

	// commitKey.GenerateKey()


	commitKeyFile, err := os.Open(commitKeyPath)

	check(err)

	scanner := bufio.NewScanner(commitKeyFile)

	// index:=0

	for scanner.Scan() {
						
		thisKeyBytes := scanner.Bytes()
		
		thisKey := PkeyG1{}

		json.Unmarshal(thisKeyBytes, &thisKey)		

		thisPointG1 := suite.G1().Point()	

		err = thisPointG1.UnmarshalBinary(thisKey.Pkey)

		check(err)

		thisPointG2 := suite.G2().Point()

		// fmt.Println(len(thisKey))

		err = thisPointG2.UnmarshalBinary(thisKey.Skey)

		check(err)

		// fmt.Println(thisKey.Id)
		// fmt.Println(thisPointG1)
		// fmt.Println(thisPointG2)

		commitKey.PKG1[thisKey.Id] = thisPointG1.Clone()

		commitKey.PKG2[thisKey.Id] = thisPointG2.Clone()	


	}

	return commitKey	

}

