package main

import (
	
	"bufio"
    "errors"
	"fmt"
	"github.com/sbinet/go-python"
	"log"
	"math"
	"net"
	"net/rpc"
    "strings"
	"strconv"
	"sync"
	"time"
	"os"
	"flag"
	"encoding/gob"

)

// Timeout for block should be more than timeout for update because nodes should be more patients for the block to come through

//Assumption: Requires node 0 to be online first. 

const (
	
	basePort        int           = 8000
	verifierIP   	string        = "127.0.0.1:"
	timeoutRPC    	time.Duration = 60 * time.Second
	timeoutUpdate 	time.Duration = 120 * time.Second 
	timeoutBlock 	time.Duration = 120 * time.Second
	timeoutPeer 	time.Duration = 5 * time.Second
	
	DEFAULT_STAKE   int 		  = 10

	PRECISION       int 		  = 4
	POLY_SIZE 		int 		  = 10

	MAX_ITERATIONS  int 		  = 100
	EPSILON 		float64 	  = 5

	SECURE_AGG  	bool 		  = true

	POISONING 	 	float64 	  = 0.0

)

type Peer int

type Vector struct {
	data []float64
}

var (

	//Input arguments
	datasetName   		string
	numberOfNodes 		int 		  

	numberOfNodeUpdates int
	myIP                string
	myPrivateIP         string
    myPort				string
    peersFileName       string


    client 				Honest
	
	allSharesReceived		chan bool
	allUpdatesReceived		chan bool
	networkBootstrapped		chan bool
	blockReceived 			chan bool
	quitRoutine 			chan bool
	portsToConnect 			[]string
	peerPorts   			[]string
    peerLookup          	map[string]int
	peerAddresses			map[int]net.TCPAddr

	//Locks
	updateLock    		sync.Mutex
	boolLock      		sync.Mutex
	convergedLock 		sync.Mutex
	peerLock			sync.Mutex

	ensureRPC      		sync.WaitGroup

	// global shared variables
    updateSent     		bool
	converged      		bool
	includePoisoned 	bool
	rndSample  			bool

	iterationCount 		= -1
	numUpdates 			= 35

	NUM_SAMPLES 		= 35
	RANDOM_SAMPLES 		= 35
	//Logging
	errLog *log.Logger = log.New(os.Stderr, "[err] ", log.Lshortfile|log.LUTC|log.Lmicroseconds)
	outLog *log.Logger = log.New(os.Stderr, "[peer] ", log.Lshortfile|log.LUTC|log.Lmicroseconds)

	//Errors
	staleError error = errors.New("Stale Update/Block")
	rpcError  error = errors.New("RPC Timeout")

)

func (s *Peer) RegisterUpdate(update Update, _ignored *bool) error {

	outLog.Printf(strconv.Itoa(client.id)+":Got update: iteration %d\n", update.Iteration)

	// we can return the chain to the guy here instead of just leaving that guy with an error
	if update.Iteration < iterationCount {
		printError("Update of previous iteration received", staleError)
		return staleError
	}

	// Process update only called by the miner nodes
	go processUpdate(update)

	return nil

}

func (s *Peer) RegisterModel(data BlockData, returnModel *BlockData) error {

	outLog.Printf(strconv.Itoa(client.id)+":Got new model, iteration %d\n", data.Iteration)

	// can return the latest block I have but there is no need yet
	*returnModel = data

	go processNewModel(data)

	return nil
	
}

func (s *Peer) RegisterPeer(peerAddress net.TCPAddr, model *Update) error {

	outLog.Printf(strconv.Itoa(client.id) + ":Registering peer:" + peerAddress.String())
	peerLock.Lock()
	peerAddresses[peerLookup[peerAddress.String()]] = peerAddress
	peerLock.Unlock()
	
	if(myPort == strconv.Itoa(basePort) && iterationCount < 0){
		networkBootstrapped <- true
	}
	
	returnUpdate := Update{SourceID: 0, 
		Iteration:iterationCount, 
		Delta: client.globalModel}

	*model = returnUpdate

	return  nil 
}

// Python init function for go-python
func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

// Error handling
func handleErrorFatal(msg string, e error) {
	if e != nil {
		errLog.Fatalf("%s, err = %s\n", msg, e.Error())
	}
}

func printError(msg string, e error) {
	if e != nil {
		errLog.Printf("%s, err = %s\n", msg, e.Error())
	}
}


func exitOnError(prefix string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s, err = %s\n", prefix, err.Error())
		os.Exit(1)
	}
}

// Parse args, read dataset and initialize separate threads for listening for updates/Blocks and sending updates

func main() {

	gob.Register(&net.TCPAddr{})
	gob.Register(&BlockData{})
	
	//Parsing arguments nodeIndex, numberOfNodes, datasetname
	numberOfNodesPtr := flag.Int("t", 0 , "The total number of nodes in the network")

	nodeNumPtr := flag.Int("i", -1 ,"The node's index in the total. Has to be greater than 0")

	datasetNamePtr := flag.String("d", "" , "The name of the dataset to be used")

    peersFileNamePtr := flag.String("f", "", "File that contains list of IP:port pairs")

    myIPPtr := flag.String("a", "", " If not local, this node's IP")

    myPrivateIPPtr := flag.String("pa", "", " If not local, this node's private IP")

    myPortPtr := flag.String("p", "", " If not local, this node's port")

    numUpdatesPtr := flag.Int("b", 35, "Number of updates to accept each round")

    rndSampleUpdatesPtr := flag.Int("ru", 35, "Number of random sampling")

    rndSamplePtr := flag.Bool("rs", false, "Random sampling")

	flag.Parse()

	nodeNum := *nodeNumPtr
	numberOfNodes = *numberOfNodesPtr
	datasetName = *datasetNamePtr
    datasetName = *datasetNamePtr
    peersFileName = *peersFileNamePtr
    myPrivateIP = *myPrivateIPPtr+":"
    myIP = *myIPPtr+":"
    myPort = *myPortPtr
    numUpdates = *numUpdatesPtr
    rndSampleUpdates := *rndSampleUpdatesPtr
    rndSample = *rndSamplePtr
    
    NUM_SAMPLES = numUpdates
    RANDOM_SAMPLES = rndSampleUpdates


	if(numberOfNodes <= 0 || nodeNum < 0 || datasetName == ""){
		flag.PrintDefaults()
		os.Exit(1)	
	}

    // getports of all other clients in the system
    peerLookup = make(map[string]int)        
    potentialPeerList := make([]net.TCPAddr, 0, numberOfNodes-1)

    // Running locally
    if (peersFileName == "") {

        myIP = "127.0.0.1:"
        myPort = strconv.Itoa(nodeNum + basePort)

        for i := 0; i < numberOfNodes; i++ {
                
            peerPort := strconv.Itoa(basePort+i)

            if peerPort == myPort {
                peerLookup[fmt.Sprintf(myIP + peerPort)] = i
                continue
            }
            
            peerPorts = append(peerPorts, peerPort)
            peerAddress, err := net.ResolveTCPAddr("tcp", fmt.Sprintf(myIP + peerPort))
            handleErrorFatal("Unable to resolve a potential peer address", err)
            potentialPeerList = append(potentialPeerList, *peerAddress)
            peerLookup[fmt.Sprintf(myIP + peerPort)] = i
        }
        
        peerAddresses = make(map[int]net.TCPAddr)

    } else if (myIP == ":" || myPort == "" || myPrivateIP == ":") {
    
        flag.PrintDefaults()
        os.Exit(1)
    
    } else {

        file, err := os.Open(peersFileName)
        handleErrorFatal("Error opening peers file", err)
        defer file.Close()

        scanner := bufio.NewScanner(file)
        nodeInList := false

        i := -1

        for scanner.Scan() {
            i++
            peerAddressStr := scanner.Text()

            if strings.Contains(peerAddressStr, myIP) && 
               strings.Contains(peerAddressStr, myPort) {
                nodeInList = true
                peerLookup[peerAddressStr] = i
                continue
            }

            peerAddress, err := net.ResolveTCPAddr("tcp", peerAddressStr)
            handleErrorFatal("Unable to resolve a potential peer address", err)
            potentialPeerList = append(potentialPeerList, *peerAddress)
            peerLookup[peerAddressStr] = i
        }

        if !nodeInList {
            handleErrorFatal("Node is not in peer list", errors.New(""))
        }

    }

    // init peer addresses list
    peerAddresses = make(map[int]net.TCPAddr)

	//Initialize a honest client
	client = Honest{id: nodeNum, blockUpdates: make([]Update, 0, 5)}

	if POISONING > 0 {

		// If your node idx is above this, you are poisoning
		poisoning_index := int(math.Ceil(float64(numberOfNodes) * (1.0 - POISONING)))
		
		outLog.Printf("Poisoning is at %d", poisoning_index)

		isPoisoning := nodeNum >= poisoning_index 
		client.initializeData(datasetName, numberOfNodes, EPSILON, isPoisoning)	
	
	} else {
		
		client.initializeData(datasetName, numberOfNodes, EPSILON, false)	
	}

	converged = false
	includePoisoned = true
	updateLock = sync.Mutex{}
	boolLock = sync.Mutex{}
	convergedLock = sync.Mutex{}
	peerLock = sync.Mutex{}

	ensureRPC = sync.WaitGroup{}
	allUpdatesReceived = make (chan bool)
	networkBootstrapped = make (chan bool)
	blockReceived = make (chan bool)
	allSharesReceived = make (chan bool)
	quitRoutine = make (chan bool)

	// Initializing RPC Server
	peer := new(Peer)
	peerServer := rpc.NewServer()
	peerServer.Register(peer)

	state := python.PyEval_SaveThread()
	
	go messageListener(peerServer, myPort)

	// announce yourself to above calculated peers. The first node in the network doesn't need to do this. He waits for an incoming peer instead. 	
	// whatever node you are you can't move on until you have announced yourself to your peers
	if(myPort != strconv.Itoa(basePort)){
		go announceToNetwork(potentialPeerList)
	}
	
	<- networkBootstrapped

	prepareForNextIteration()
	messageSender()
	python.PyEval_RestoreThread(state)

}


// peers announce themselves to all other nodes when they come into the system 

func announceToNetwork(peerList []net.TCPAddr){

	// change from everything from ports to net.TCPAddr
	outLog.Printf(strconv.Itoa(client.id)+":Announcing myself to network")
	myAddress, err := net.ResolveTCPAddr("tcp", fmt.Sprintf(myIP + myPort))
	exitOnError("Resolve own address", err)

	for _, address := range peerList {
        outLog.Printf(strconv.Itoa(client.id)+":Calling %s", address)
		callRegisterPeerRPC(*myAddress, address)		
	}

	// if havent been able to find a peer then I WILL DIE
	if(len(peerAddresses) == 0){
		outLog.Printf(strconv.Itoa(client.id)+":No peers to connect to. I WILL DIE")
		os.Exit(1)
	}
    
    outLog.Printf(strconv.Itoa(client.id)+":Bootstrapped Network. Calling signal")
	networkBootstrapped <- true
	outLog.Printf(strconv.Itoa(client.id)+":Bootstrapped Network. Signal called")

}

func callRegisterPeerRPC(myAddress net.TCPAddr, peerAddress net.TCPAddr) {

	
	outLog.Printf(strconv.Itoa(client.id)+":Inside callRegisterRPC")
	// var chain Blockchain
	model := Update{}
	c := make(chan error)

	outLog.Printf(strconv.Itoa(client.id)+"Making RPC call")
	conn, err := rpc.Dial("tcp", peerAddress.String()) 
	printError("Peer offline.Couldn't connect to peer: " + peerAddress.String(), err)

	// Wait for node 0 to come online if it already hasn't"

	if ( (peerLookup[peerAddress.String()] == 0) && err!=nil) {
		
		outLog.Printf("Waiting for node 0 to come online")		

		for (err!=nil) {
			time.Sleep(1000 * time.Millisecond)
			conn, err = rpc.Dial("tcp", peerAddress.String()) 
			printError("Peer offline.Couldn't connect to peer: " + peerAddress.String(), err)
		}

	} 
	
	if(err == nil){

		outLog.Printf(strconv.Itoa(client.id)+"RPC dial successful")

		defer conn.Close()	
		outLog.Printf(strconv.Itoa(client.id)+":Calling RPC:"+ peerAddress.String())
		go func() { c <- conn.Call("Peer.RegisterPeer", myAddress, &model) }()
		outLog.Printf(strconv.Itoa(client.id)+":RPC called"+ peerAddress.String())
		select {

		case err = <-c:

			if(err == nil){
				
				outLog.Printf(strconv.Itoa(client.id)+":Announced myself to a fellow peer at port. Got latest chain")
				
				//Add peer
				peerLock.Lock()
				peerAddresses[peerLookup[peerAddress.String()]] = peerAddress
				peerLock.Unlock()

				client.globalModel = model.Delta

			}

			// use err and result
		case <-time.After(timeoutPeer):

			outLog.Printf(strconv.Itoa(client.id)+":Couldn't get response from peer: "+ peerAddress.String())
		}
	}	

}

// At the start of each iteration, this function is called to reset shared global variables
// based on whether you are a verifier or not.

func prepareForNextIteration() {

	convergedLock.Lock()
	
	if converged {
		
		convergedLock.Unlock()
		time.Sleep(1000 * time.Millisecond)
		os.Exit(1)
	
	} else {

		if iterationCount > MAX_ITERATIONS {
			
			outLog.Println("Reached the max iterations!")
			os.Exit(1)	
		
		}
	}

	updateLock.Lock()
	client.flushUpdates()
	updateLock.Unlock()		

	convergedLock.Unlock()		
	boolLock.Lock()
	
	iterationCount++
	outLog.Printf("Moving on to next iteration %d", iterationCount)

	numberOfNodeUpdates = (numberOfNodes)/8

	if amLeader() {
		go startUpdateDeadlineTimer(iterationCount)
	} else {
		updateSent = false
	}
	
	boolLock.Unlock()

	portsToConnect = make([]string, len(peerPorts))
	copy(portsToConnect, peerPorts)

}

// Thread that listens for incoming RPC Calls

func messageListener(peerServer *rpc.Server, port string) {

	l, e := net.Listen("tcp", myPrivateIP + port)
	exitOnError("listen error", e)
	defer l.Close()

	outLog.Printf(strconv.Itoa(client.id)+":Peer started. Receiving on %s\n", port)

	for {
		conn, _ := l.Accept()
		go peerServer.ServeConn(conn)
	}

}

// go routine to process the update received by miner nodes
func processUpdate(update Update) {

	outLog.Printf(strconv.Itoa(client.id)+":Got update for %d, I am at %d\n", update.Iteration, iterationCount)

	for update.Iteration > iterationCount {
		outLog.Printf(strconv.Itoa(client.id)+":Blocking for stale update. Update for %d, I am at %d\n", update.Iteration, iterationCount)
		time.Sleep(2000 * time.Millisecond)
	}

	// Might get an update while I am in the announceToNetwork phase and when I come out of it the update becomes redundant
	if ((iterationCount == update.Iteration)) {

		updateLock.Lock()
		numberOfUpdates := client.addBlockUpdate(update)
		updateLock.Unlock()

		outLog.Printf("As aggregator, I expect %d updates, I have gotten %d", numberOfNodeUpdates, numberOfUpdates)

		if (numberOfUpdates == NUM_SAMPLES) {			
			
			if (rndSample) {
				client.sampleUpdates(RANDOM_SAMPLES)
			}

			outLog.Printf(strconv.Itoa(client.id)+":All updates for iteration %d received. Notifying channel.", iterationCount)	
			allUpdatesReceived <- true

		}	

		// if POISONING > 0 {

		// 	if (numberOfUpdates >= (numberOfNodes - 1)) {			
			
		// 		client.sampleUpdates(numberOfNodeUpdates)
		// 		outLog.Printf(strconv.Itoa(client.id)+":All updates for iteration %d received. Notifying channel.", iterationCount)	
		// 		allUpdatesReceived <- true
		// 	}

		// }else{

		// 	if (numberOfUpdates >= (numberOfNodes - 1))  {

		// 		client.sampleUpdates(numberOfNodeUpdates)			
		// 		outLog.Printf(strconv.Itoa(client.id)+":Half updates for iteration %d received. Notifying channel.", iterationCount)	
		// 		allUpdatesReceived <- true
		// 	}

		// }

		//send signal to start sending Block if all updates Received. Changed this from numVanilla stuff
			
	
	}

}

// // For all non-miners, accept the block
func processNewModel(data BlockData) {

	if (data.Iteration < iterationCount || iterationCount < 0) {
		return
	}

	outLog.Printf(strconv.Itoa(client.id)+":Got new model, iteration %d\n", data.Iteration)

	if (!updateSent && data.Iteration == iterationCount) {
		updateSent = true
		outLog.Printf("Releasing worker on iteration %d", iterationCount)
	}

	client.globalModel = data.GlobalW

	convergedLock.Lock()
	converged = client.checkConvergence(iterationCount)
	convergedLock.Unlock()

	prepareForNextIteration()

}


// Server broadcasts the new model of this iteration to all peers
func sendModel(data BlockData) {	

	outLog.Printf(strconv.Itoa(client.id)+":Sending model of iteration: %d\n", data.Iteration)

	// create a thread for separate calling
	peerLock.Lock()

	ensureRPC.Add(len(peerAddresses))
	
	for _, address := range peerAddresses {
		go callRegisterModelRPC(data, address)
	}
	
	//check for convergence, wait for RPC calls to return and move to the new iteration

	ensureRPC.Wait()

	// ensureRPCCallsReturn()
	peerLock.Unlock()

	outLog.Printf(strconv.Itoa(client.id)+":RPC calls successfully returned. Iteration: %d", iterationCount)

	client.globalModel = data.GlobalW

	convergedLock.Lock()
	converged = client.checkConvergence(iterationCount)
	convergedLock.Unlock()

	outLog.Printf(strconv.Itoa(client.id)+":Preparing for next Iteration. Current Iteration: %d", iterationCount)

	prepareForNextIteration()
		
}

// RPC call to send block to one peer
func callRegisterModelRPC(data BlockData, peerAddress net.TCPAddr) {

	defer ensureRPC.Done()
	var returnBlock BlockData
	c := make(chan error)

	conn, er := rpc.Dial("tcp", peerAddress.String()) 
	printError("rpc Dial", er)

	// outLog.Printf("Sending Global Model: %v", data.GlobalW)

	if(er==nil){

		defer conn.Close()
		go func() { c <- conn.Call("Peer.RegisterModel", data, &returnBlock) }()
		select {
		case err := <-c:

			outLog.Printf(strconv.Itoa(client.id)+":Model sent to peer successful. Peer: " + peerAddress.String() + " Iteration: %d", client.iteration)
			printError("Error in sending block", err)
			// ensureRPC <- true

			// use err and result
		case <-time.After(timeoutRPC):

			// On timeout delete peer because its unresponsive
			outLog.Printf("Timeout. Sending Block.")
			delete(peerAddresses, peerLookup[peerAddress.String()])
		}

	}else{

		delete(peerAddresses, peerLookup[peerAddress.String()])
		outLog.Printf(strconv.Itoa(client.id)+":Peer Unresponsive. Removed Peer:" + peerAddress.String())

	}	

}

// Main sending thread. Checks if you are a non-verifier in the current itearation 
// Sends update if thats the case.

func messageSender() {

	if (myPort == strconv.Itoa(basePort)) {
		// wait forever
		select {}
	}

	for {

		boolLock.Lock()

		if !updateSent {
			outLog.Printf(strconv.Itoa(client.id)+":Computing Update\n")
			client.computeUpdate(iterationCount)
			sendUpdateToServer()
			updateSent = true

		}

		if updateSent {
			boolLock.Unlock()
			time.Sleep(1000 * time.Millisecond)
		}
		
	}
}

func sendUpdateToServer() {

	var ign bool
	c := make(chan error)

	leader := getLeaderAddress()

	conn, err := rpc.Dial("tcp", leader)
	printError("Unable to connect to server", err)
			
	if (err == nil) {
				
		defer conn.Close()
		outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Miner. Sending Update, Iteration:%d\n", client.update.Iteration)
		go func() { c <- conn.Call("Peer.RegisterUpdate", client.update, &ign) }()
	
		select {
			case err := <-c:
				
				printError("Error in sending update", err)
				if(err==nil){
					outLog.Printf(strconv.Itoa(client.id)+":Update sent. Iteration:%d\n", client.update.Iteration)
				}

				if(err==staleError){
					outLog.Printf(strconv.Itoa(client.id)+"Stale error:Update mined. Iteration:%d\n", client.update.Iteration)
				}

				// use err and result
			case <-time.After(timeoutRPC):
				outLog.Printf(strconv.Itoa(client.id)+":RPC Call timed out.")

		}
		
	} else {
		
		outLog.Printf("GOT SERVER ERROR")
		os.Exit(1)
	}

}

func amLeader() bool {
	port, _ := strconv.Atoi(myPort) 
	return port == basePort
}

func getLeaderAddress() string {
	leader := peerAddresses[0]
	return leader.String()
}
	

// Timer started by the miner to set a deadline for receiving updates
func startUpdateDeadlineTimer(timerForIteration int){
	
	select {
		
		case <- allUpdatesReceived:
			outLog.Printf(strconv.Itoa(client.id)+":All Updates Received for timer on %d. I am at %d. Preparing to send block..", 
				timerForIteration, iterationCount)

		case <- time.After(timeoutUpdate):
			outLog.Printf(strconv.Itoa(client.id)+":Timeout. Didn't receive expected number of updates. Preparing to send block. Iteration: %d..", iterationCount)
	
	}
	
	if (timerForIteration == iterationCount) {
			
		if (len(client.blockUpdates) > 0) {

			client.sampleUpdates(numUpdates)
	
			modelToSend, err := client.createNewModel(iterationCount)
			
			printError("Iteration: " + strconv.Itoa(iterationCount), err)
			
			if (err == nil) {
				sendModel(modelToSend)
			}

		} else {

			outLog.Printf("Timer is for %d", timerForIteration)
			outLog.Printf("I am on %d", iterationCount)

			outLog.Printf(strconv.Itoa(client.id)+":Received no updates from peers. I WILL DIE")
			os.Exit(1)
		} 

	}
}