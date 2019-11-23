package main

import(
	"strconv"
	"time"
	"github.com/sbinet/go-python"
	"runtime"
	"encoding/binary"
	"math/rand"
	"math"
	"sort"

	// "bytes"
	// "math"
)

var (

	pyKRUMFunc 		*python.PyObject
)

type KRUMValidator struct {

	UpdateList  	[]Update
	SampledUpdates  []Update	
	AcceptedList 	[]int
	NumAdversaries	float64 // fixed at 0.5

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

	poisoning_index := int(math.Ceil(float64(numberOfNodes) * (1.0 - POISONING)))

	isPeerPoisoning := peerId > poisoning_index	

	if (isPoisoning && isPeerPoisoning){		
		accepted = true
		return accepted	
	}

	for i := 0; i < len(krumval.AcceptedList); i++ {

		acceptedIdx := krumval.AcceptedList[i]

		if krumval.UpdateList[acceptedIdx].SourceID == peerId  {
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

	runningDeltas := make([][]float64, len(krum.UpdateList))

	for i := 0; i < len(krum.UpdateList); i++ {
		runningDeltas[i] = krum.UpdateList[i].NoisedDelta
	}

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
	
	if !collectingUpdates{
		krumval.UpdateList = krumval.UpdateList[:0]
		krumval.AcceptedList = krumval.AcceptedList[:0]
	}

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

				collectingUpdates = false

				sort.Slice(krum.UpdateList, func(i, j int) bool {
	  				return krum.UpdateList[i].SourceID < krum.UpdateList[j].SourceID
				})

				if RAND_SAMPLE {
					krum.sampleUpdates(NUM_SAMPLES)					
				}
				
				krum.computeScores()


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

		// if update.Iteration == iterationCount{

		// 	krumLock.Lock()

		// 	if !collectingUpdates{

		// 		krum.flushCollectedUpdates()
		// 		collectingUpdates = true
		// 	}

		// 	krumLock.Unlock()

		// }

		if update.Iteration == iterationCount{

			for !collectingUpdates {
				
				outLog.Printf(strconv.Itoa(client.id)+"Collecting Updates not true. Update for %d, I am at %d\n", update.Iteration, iterationCount)				

				time.Sleep(100 * time.Millisecond)
			
			}


			// krumLock.Lock()

			// if !collectingUpdates{				
			// 	krum.flushCollectedUpdates()
			// 	collectingUpdates = true
			// }

			// krumLock.Unlock()
		
		}
	
	}	

	// TODO: set up acquiring a lock here
	krumLock.Lock()
	// peerId := 0	

	if (collectingUpdates) {

		// peerId = len(krum.UpdateList)

		krum.UpdateList = append(krum.UpdateList, update)
		outLog.Printf(strconv.Itoa(client.id)+"Inside collectingUpdates. Update for %d, I am at %d\n", update.Iteration, iterationCount)
		outLog.Printf(strconv.Itoa(client.id)+"List length %d", len(krum.UpdateList))
		//TODO: Declare UpdateThresh	

		if (len(krum.UpdateList) == KRUM_UPDATETHRESH){
			
			outLog.Printf(strconv.Itoa(client.id)+"Reached KRUM THRESH at %d\n", iterationCount)
			
			krumReceived <- true
			outLog.Printf(strconv.Itoa(client.id)+"Crossed Received %d\n", iterationCount)
			
			
			collectingUpdates = false

			sort.Slice(krum.UpdateList, func(i, j int) bool {
  				return krum.UpdateList[i].SourceID < krum.UpdateList[j].SourceID
			})

			if (RAND_SAMPLE){
				krum.sampleUpdates(NUM_SAMPLES)
			}

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

		if krum.checkIfAccepted(update.SourceID){

			outLog.Printf("Accepting update!")
			
			poisoning_index := int(math.Ceil(float64(numberOfNodes) * (1.0 - POISONING)))
			isPoisoning := update.SourceID > poisoning_index

			if isPoisoning {
				outLog.Printf("Accepting poisoned update!")
			}

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

// cSample updates for the poisoning case
func (krumval *KRUMValidator) sampleUpdates(numUpdates int) {

	r := rand.New(rand.NewSource(int64(iterationCount)))
	selectedUpdates := make([]Update, numUpdates)
	perm := r.Perm(len(krumval.UpdateList))
	
	for i, randIndex := range perm {

		selectedUpdates[i] = krumval.UpdateList[randIndex]
		if i == (numUpdates-1) {
			break
		}
	
	}	

	krumval.UpdateList = selectedUpdates

	outLog.Printf("Number of updates sampled:%s", len(krumval.UpdateList))
	// outLog.Printf("Indexes selected:%s", perm[:numUpdates])

}

