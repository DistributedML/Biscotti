package main

import(
	"strconv"
	"time"
	"github.com/sbinet/go-python"
	"runtime"
	"encoding/binary"
	// "bytes"
	// "math"
)

var (

	pyKRUMFunc 		*python.PyObject
)

type KRUMValidator struct {

	UpdateList  	[][]float64
	AcceptedList 	[]int
	NumAdversaries	float64

}

func (krumval *KRUMValidator) initialize(){

	if useTorch {

		pyKRUMFunc = pyTorchModule.GetAttrString("krum")

	}else {

		pyKRUMFunc = pyRoniModule.GetAttrString("krum")
		outLog.Printf("Krum function is:%s", pyKRUMFunc)

	}

}


func (krumval *KRUMValidator) checkIfAccepted(peerId int) bool {
	
	accepted := false

	for i := 0; i < len(krumval.AcceptedList); i++ {
		
		if krumval.AcceptedList[i] == peerId  {			
			accepted = true
			return accepted
		}

	}

	return accepted

}


//TODO: Replace with python call to krum
func (krumval *KRUMValidator) computeScores(){
	
	// numUpdatesRec := int(len(krumval.UpdateList))
	// numberToReject := int(krumval.NumAdversaries*float64(numUpdatesRec))
	// numberToAccept := numberToReject - numUpdatesRec

	runningDeltas := krumval.UpdateList
	krum.AcceptedList = krumval.getTopKRUMIndex(runningDeltas)
	// numInBlock = numberOfNodes/8
	// krum.AcceptedList = krum.AcceptedList[:numberOfNodes]
	outLog.Printf("List of accepted people:%s", krum.AcceptedList)


	// for i := 0; i < len(acceptedListFloats); i++ {		
	// 	krumval.AcceptedList = append(krumval.AcceptedList, int(acceptedListFloats[i]))
	// }
}

func (krumval *KRUMValidator) getTopKRUMIndex(deltas [][]float64) []int{

	//Making Python call to KRUM
	outLog.Println("Acquiring Python Lock...")
	runtime.LockOSThread()

	_gstate := python.PyGILState_Ensure()
	outLog.Println("Acquired Python Lock")

	numUpdates := len(deltas)
	adversaryCount := int(krumval.NumAdversaries*float64(numUpdates))

	pyDeltas := python.PyList_New(len(deltas))
	// updateArray := python.PyList_New(len(truthModel))

	// TODO: Create a two d array

	for i := 0; i < len(deltas); i++ {
		thisDelta := python.PyList_New(len(deltas[i]))

		for j := 0; j < len(deltas[i]); j++ {
			python.PyList_SetItem(thisDelta, j, python.PyFloat_FromDouble(deltas[i][j]))
		}

		python.PyList_SetItem(pyDeltas,i,thisDelta)
	}

	pyAdversaries := python.PyInt_FromLong(adversaryCount)
	
	var result *python.PyObject
	result = pyKRUMFunc.CallFunction(pyDeltas, pyAdversaries)
	outLog.Printf("Result from krum python:%s", result)

	// Convert the resulting array to a go byte array
	pyByteArray := python.PyByteArray_FromObject(result)
	goByteArray := python.PyByteArray_AsBytes(pyByteArray)

	outLog.Println("Go Byte Array:%s", len(goByteArray) )	
		outLog.Println("Go Byte Array:%s", goByteArray )	

	// acceptedIdxs := binary.BigEndian.Uint64(goByteArray[s])

	var acceptedIdxs []int
	var acceptedIdx int
	size := len(goByteArray) / 8

	// buf := bytes.NewBuffer(goByteArray) // b is []byte	

	for i := 0; i < size; i++ {
		currIndex := i * 8
		temp := binary.LittleEndian.Uint64(goByteArray[currIndex : currIndex+8])
		// buf := goByteArray[currIndex : currIndex+8]
		// outLog.Println("Buffer:%s", buf )
		// aFloat64, _ := binary.ReadVarint(buf)
		acceptedIdx = int(temp) 	

		// _ = binary.Read(bytes.NewReader(buf), binary.BigEndian, &aFloat)		
		acceptedIdxs = append(acceptedIdxs, acceptedIdx)
	}

	outLog.Println("Outside KRUM")	
	python.PyGILState_Release(_gstate)	
	outLog.Println("Released Python Lock")

	return acceptedIdxs

}

// Empty the validator
func (krumval *KRUMValidator) flushCollectedUpdates(){
	
	krumval.UpdateList = krumval.UpdateList[:0]
	krumval.AcceptedList = krumval.AcceptedList[:0]

}

func startKRUMDeadlineTimer(timerForIteration int){
	
	outLog.Printf(strconv.Itoa(client.id)+"Starting KRUM deadline timer %d\n", iterationCount)		
	
	select{
		
		case <- krumReceived:
			
			outLog.Printf(strconv.Itoa(client.id)+"Required updates received for iteration: %d", timerForIteration)

			if (timerForIteration == iterationCount) {
				outLog.Printf(strconv.Itoa(client.id)+"Evaluating KRUM for iteration %d", iterationCount)
			}

		case <-time.After(timeoutKRUM):
		
			krumLock.Lock()

			if (timerForIteration == iterationCount) {
				
				outLog.Printf(strconv.Itoa(client.id)+":Timeout. Going ahead with KRUM for iteration", timerForIteration)		
							
				krum.computeScores()

				collectingUpdates = false

				for i := 0; i < len(krum.UpdateList); i++ {
				
					krumAccepted <- true			
					
				}

			}

			krumLock.Unlock()	
	}

}


func (s *Peer) VerifyUpdateKRUM(update Update, signature *[]byte) error {

	outLog.Printf(strconv.Itoa(client.id)+":Got KRUM message, iteration %d\n", update.Iteration)

	if (update.Iteration < iterationCount) {

		printError("Update of previous iteration received", staleError)
		return staleError

	}

	if (update.Iteration > iterationCount) {
		
		for update.Iteration > iterationCount {
			outLog.Printf(strconv.Itoa(client.id)+":Blocking for stale update. Update for %d, I am at %d\n", update.Iteration, iterationCount)
			time.Sleep(2000 * time.Millisecond)
		}
	
	}	

	// TODO: set up acquiring a lock here
	krumLock.Lock()
	peerId := 0	

	if collectingUpdates {

		peerId = len(krum.UpdateList)

		krum.UpdateList = append(krum.UpdateList, update.NoisedDelta)

		//TODO: Declare UpdateThresh	

		if (len(krum.UpdateList) == KRUM_UPDATETHRESH){
			
			outLog.Printf(strconv.Itoa(client.id)+"Reached KRUM THRESH at %d\n", iterationCount)
			
			krumReceived <- true
			outLog.Printf(strconv.Itoa(client.id)+"Crossed Received %d\n", iterationCount)
			
			
			collectingUpdates = false

			krum.computeScores()

			outLog.Printf(strconv.Itoa(client.id)+"Crossed Accepted %d\n", iterationCount)

			for i := 0; i < (KRUM_UPDATETHRESH-1); i++ {
			
				krumAccepted <- true								
				
			}

			krumLock.Unlock()

			outLog.Printf("KRUM Processing Complete")

			// shouldn't be going out. return here

		}else{
			
			krumLock.Unlock()

			<- krumAccepted				

		}

		if krum.checkIfAccepted(peerId){

			outLog.Printf("Accepting update!")
			updateCommitment := update.Commitment
			(*signature) = SchnorrSign(updateCommitment, client.Keys.Skey)
			return nil
		
		}else{

			outLog.Printf("Rejecting update!")
			return updateError
		
		}		
	
	}

	krumLock.Unlock()
	printError("Not collectingUpdates anymore", staleError)
	return staleError	
}