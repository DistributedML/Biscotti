package main

import (
	"errors"
	"fmt"
	"github.com/sbinet/go-python"
	"log"
	"net"
	"net/rpc"
	"strconv"
	"sync"
	"time"
	"os"
	"flag"
	"encoding/gob"
)

// TODO: Figure out best values of timeouts to prevent unnecessary delays. Timeout for RegisterPeer< timeout for other two RPC's
// Register Peer RPC is in and out so it must be equal to some times the RTT. TimeoutRPC for the other 2 remains the same
// verifyUpdate, registerBlock might block client due to being behind in another iteration therefore its significantly larger. This is no longer true so can experiment with lower timeouts
// timeoutBlock larger than timmeout update because verifier has to collect updates from everyone before sending out block so the block might come later
// than the . Do the values need to be this large? Remains open to debate.

//Assumption: Requires node 0 to be online first. Need to move away from this

const (
	basePort     	int           = 8000
	myIP         	string        = "127.0.0.1:"
	verifierIP   	string        = "127.0.0.1:"
	timeoutRPC    	time.Duration = 10000000000
	numVerifiers 	int           = 1
	timeoutUpdate 	time.Duration = 15000000000  
	timeoutBlock 	time.Duration = 16000000000  
	timeoutPeer 	time.Duration = 1000000000
)

type Peer int

var (

	//Input arguments
	datasetName   		string
	numberOfNodes 		int
	myPort				string

	client 				Honest

	
	allUpdatesReceived	chan bool
	networkBootstrapped	chan bool
	blockReceived 		chan bool
	portsToConnect 		[]string
	peerPorts   		[]string
	peerAddresses		map[string]net.TCPAddr

	//Locks
	updateLock    		sync.Mutex
	boolLock      		sync.Mutex
	convergedLock 		sync.Mutex
	peerLock			sync.Mutex
	blockChainLock		sync.Mutex

	ensureRPC      		sync.WaitGroup

	// global shared variables
	updateSent     		bool
	converged      		bool
	verifier       		bool
	iterationCount 		= -1

	//Logging
	errLog *log.Logger = log.New(os.Stderr, "[err] ", log.Lshortfile|log.LUTC|log.Lmicroseconds)
	outLog *log.Logger = log.New(os.Stderr, "[peer] ", log.Lshortfile|log.LUTC|log.Lmicroseconds)

	//Errors
	staleError error = errors.New("Stale Update/Block")
)

// Python init function for go-python
func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

// RPC CALLS

// The peer receives an update from another peer if its a verifier in that round.
// The verifier peer takes in the update and returns immediately.
// It calls a separate go-routine for collecting updates and sending updates when all updates have been collected
// Returns:
// - StaleError if its an update for a preceding round.

func (s *Peer) VerifyUpdate(update Update, _ignored *bool) error {

	outLog.Printf("Got update message, iteration %d\n", update.Iteration)

	if update.Iteration < iterationCount {
		handleErrorFatal("Update of previous iteration received", staleError)
		return staleError
	}

	go processUpdate(update)

	return nil

}

// The peer receives a block from the verifier of that round.
// It takes in the block and returns immediately.
// It calls a separate go-routine for appending the block as part of its chain
// Returns:
// - staleError if its an block for a preceding round.

func (s *Peer) RegisterBlock(block Block, returnBlock *Block) error {

	outLog.Printf("Got block message, iteration %d\n", block.Data.Iteration)

	*returnBlock = block

	// boolLock.Lock()

	// Lock to ensure that iteration count doesn't change until I have appended block
	outLog.Printf("Acquiring bool lock")
	boolLock.Lock()
	outLog.Printf("Acquired bool lock")

	if ((block.Data.Iteration < iterationCount) || client.hasBlock(block.Data.Iteration)) {
		
		boolLock.Unlock()
		outLog.Printf("Bool lock released")		

		better := client.evaluateBlockQuality(block) // check equality and some measure of 	

		if(better){
			
			// TODO: If I receive a better block than my current one. Then I replace my block with this one.
			// I request for all the next Blocks. I will also need to advertise new block or not?
			// go callRequestChainRPC(same conn) // returns whole chain. Is it longer than mine?
			// go evaluateReceivedChain() // if chain longer than mine and checks out replace mine with his
			
			if(block.Data.Iteration == len(client.bc.Blocks) - 2){
				client.replaceBlock(block, block.Data.Iteration)
				outLog.Printf("Received better  block")
				return nil
			}

		
		}else{
			
			returnBlock = client.bc.getBlock(block.Data.Iteration)						
			outLog.Printf("Equal block")
			return staleError
		
		}

		// handleErrorFatal("Block of previous iteration received", staleError)
	}


	outLog.Printf("Sending to channel")
	// if not empty send signal to channel
	if(len(block.Data.Deltas) != 0 && !verifier) {
		blockReceived <- true
	}

	outLog.Printf("Sent to channel")
	go addBlockToChain(block)
	outLog.Printf("Returning")


	return nil

}

// when peer calls does register block.
// When register block returns, return chain.
// Get chain from all peers. Accept the longest one.
// Set iteration count based on that

// Register a fellow node as a peer.
// Returns:
// 	-nil if peer added

func (s *Peer) RegisterPeer(peerAddress net.TCPAddr, chain *Blockchain) error {

	outLog.Printf("Registering peer:" + peerAddress.String())
	peerLock.Lock()
	peerAddresses[peerAddress.String()] = peerAddress
	peerLock.Unlock()
	if(myPort == strconv.Itoa(basePort)){
		networkBootstrapped <- true
	}
	chain = client.bc
	return  nil 
}


// Basic check to see if you are the verifier in the next round

func amVerifier(nodeNum int) bool {

	//TODO: THIS WILL CHANGE AS OUR VRF APPROACH MATURES.
	if (iterationCount % numberOfNodes) == client.id {
		return true
	} else {
		return false
	}

}

// Dummy placeholder VRF function

func VRF(iterationCount int) []string {

	// TODO: THIS WILL CHANGE AS THE VRF IMPLEMENTATION CHANGES
	verifiers := make([]string, numVerifiers)
	verifiers[0] = strconv.Itoa(basePort + (iterationCount % numberOfNodes))
	return verifiers

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
	gob.Register(&Blockchain{})
	
	//Parsing arguments nodeIndex, numberOfNodes, datasetname
	numberOfNodesPtr := flag.Int("t", 0 , "The total number of nodes in the network")

	nodeNumPtr := flag.Int("i", -1 ,"The node's index in the total. Has to be greater than 0")

	datasetNamePtr := flag.String("d", "" , "The name of the dataset to be used")

	flag.Parse()

	nodeNum := *nodeNumPtr
	numberOfNodes = *numberOfNodesPtr
	datasetName = *datasetNamePtr

	if(numberOfNodes <= 0 || nodeNum < 0 || datasetName == ""){
		flag.PrintDefaults()
		os.Exit(1)	
	}

	

	// getports of all other clients in the system
	myPort = strconv.Itoa(nodeNum + basePort)
	potentialPeerList := make([]net.TCPAddr, 0, numberOfNodes-1)

	for i := 0; i < numberOfNodes; i++ {
		if strconv.Itoa(basePort+i) == myPort {
			continue
		}
		peerPort := strconv.Itoa(basePort+i)
		peerPorts = append(peerPorts, peerPort)
		peerAddress, err := net.ResolveTCPAddr("tcp", fmt.Sprintf(myIP + peerPort))
		handleErrorFatal("Unable to resolve a potentail peer address", err)
		potentialPeerList = append(potentialPeerList, *peerAddress)
	}
	peerAddresses = make(map[string]net.TCPAddr)

	//Initialize a honest client
	client = Honest{id: nodeNum, blockUpdates: make([]Update, 0, 5)}

	
	// Reading data and declaring some global locks to be used later
	client.initializeData(datasetName, numberOfNodes)
	

	converged = false
	verifier = false	
	updateLock = sync.Mutex{}
	boolLock = sync.Mutex{}
	convergedLock = sync.Mutex{}
	peerLock = sync.Mutex{}
	blockChainLock = sync.Mutex{}

	ensureRPC = sync.WaitGroup{}
	allUpdatesReceived = make (chan bool)
	networkBootstrapped = make (chan bool)
	blockReceived = make (chan bool)



	// Initializing RPC Server
	peer := new(Peer)
	peerServer := rpc.NewServer()
	peerServer.Register(peer)

	state := python.PyEval_SaveThread()
	

	go messageListener(peerServer, myPort)

	// announce yourself to above calculated peers. The first node in the network doesn't need to do this. He waits for an incoming peer instead. 	
	if(myPort != strconv.Itoa(basePort)){
		announceToNetwork(potentialPeerList)
	}else{
		<- networkBootstrapped
	}

	prepareForNextIteration()
	
	messageSender(peerPorts)

	python.PyEval_RestoreThread(state)


}



// peers announce themselves to all other nodes when they come into the system 

func announceToNetwork(peerList []net.TCPAddr){

	// change from everything from ports to net.TCPAddr
	outLog.Printf("Announcing myself to network")
	myAddress, err := net.ResolveTCPAddr("tcp", fmt.Sprintf(myIP + myPort))
	exitOnError("Resolve own address", err)

	for _, address := range peerList{
		callRegisterPeerRPC(*myAddress, address)		
	}

}

func callRegisterPeerRPC(myAddress net.TCPAddr, peerAddress net.TCPAddr) {

	var chain Blockchain
	c := make(chan error)

	conn, err := rpc.Dial("tcp", peerAddress.String()) 
	printError("Peer offline.Couldn't connect to peer: " + peerAddress.String(), err)
	
	// will need to change this to receive the chain

	if(err == nil){

		defer conn.Close()	
		outLog.Printf("Calling RPC:"+ peerAddress.String())
		go func() { c <- conn.Call("Peer.RegisterPeer", myAddress, &chain) }()
		outLog.Printf("RPC called"+ peerAddress.String())
		select {

		case err = <-c:

			if(err == nil){
				
				outLog.Printf("Announced myself to a fellow peer at port. Got lastest chain")
				
				//Add peer
				peerLock.Lock()
				peerAddresses[peerAddress.String()] = peerAddress
				peerLock.Unlock()

				//Check the chain and see if its the longest one. If longer replace it with mine
				if(len(chain.Blocks) > len(client.bc.Blocks)){
					client.replaceChain(chain)
				}

			}


			// use err and result
		case <-time.After(timeoutPeer):

			outLog.Printf("Couldn't get response from peer: "+ peerAddress.String())
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
		client.bc.PrintChain()
		os.Exit(1)
	}

	convergedLock.Unlock()

	outLog.Printf("Acquiring bool lock")
	boolLock.Lock()
	outLog.Printf("Acquired bool lock")

	if verifier {
		updateLock.Lock()
		client.flushUpdates(numberOfNodes)
		updateLock.Unlock()
	}

	iterationCount++

	verifier = amVerifier(client.id)

	if verifier {
		outLog.Printf("I am verifier. IterationCount:%d", iterationCount)
		go startUpdateDeadlineTimer() //start timer for receiving updates
		updateSent = true
	} else {
		outLog.Printf("I am not verifier IterationCount:%d", iterationCount)
		updateSent = false
	}

	boolLock.Unlock()
	outLog.Printf("Bool lock released")

	portsToConnect = make([]string, len(peerPorts))
	copy(portsToConnect, peerPorts)

}

// Thread that listens for incoming RPC Calls

func messageListener(peerServer *rpc.Server, port string) {

	l, e := net.Listen("tcp", myIP+port)
	handleErrorFatal("listen error", e)

	outLog.Printf("Peer started. Receiving on %s\n", port)

	for {
		conn, _ := l.Accept()
		outLog.Printf("Accepted new Connection")
		go peerServer.ServeConn(conn)
	}

}

// go routine to process the update received by non verifying nodes

func processUpdate(update Update) {

	for update.Iteration > iterationCount {
		outLog.Printf("Blocking. Got update for %d, I am at %d\n", update.Iteration, iterationCount)
		time.Sleep(5000 * time.Millisecond)
	}

	updateLock.Lock()
	numberOfUpdates := client.addBlockUpdate(update)
	updateLock.Unlock()

	//send signal to start sending Block if all updates Received
	if numberOfUpdates == (numberOfNodes - 1) {			
		allUpdatesReceived <- true 		 
	}


}

// Verifier broadcasts the block of this iteration to all peers

func sendBlock(block Block) {	

	outLog.Printf("Sending block. Iteration: %d\n", block.Data.Iteration)

	// create a thread for separate calling
	peerLock.Lock()

	ensureRPC.Add(len(peerAddresses))
	
	for _, address := range peerAddresses {
		go callRegisterBlockRPC(block, address)
	}
	
	//check for convergence, wait for RPC calls to return and move to the new iteration

	ensureRPC.Wait()

	// ensureRPCCallsReturn()
	peerLock.Unlock()

	// You can only move to the next iteration by sending a block if you were the verifier for that iteration or if you are proposing an empty block

	outLog.Printf("RPC calls successfully returned. Iteration: %d", iterationCount)

	// if(block.Data.Iteration == iterationCount && (verifier || len(block.Data.Deltas) == 0 )){

	convergedLock.Lock()
	converged = client.checkConvergence()
	convergedLock.Unlock()

	outLog.Printf("Preparing for next Iteration. Current Iteration: %d", iterationCount)

	prepareForNextIteration()

	// }
		

}

// output from channel to ensure all RPC calls to broadcast block are successful

// func ensureRPCCallsReturn() {

// 	for i := 0; i < len(peerAddresses); i++ {
// 		<-ensureRPC
// 	}

// }

// RPC call to send block to one peer

func callRegisterBlockRPC(block Block, peerAddress net.TCPAddr) {

	defer ensureRPC.Done()
	var returnBlock Block
	c := make(chan error)

	conn, er := rpc.Dial("tcp", peerAddress.String()) 
	printError("rpc Dial", er)

	if(er==nil){

		defer conn.Close()
		go func() { c <- conn.Call("Peer.RegisterBlock", block, &returnBlock) }()
		select {
		case err := <-c:

			outLog.Printf("Block sent to peer successful")
			printError("Error in sending block", err)
			// ensureRPC <- true

			// use err and result
		case <-time.After(timeoutRPC):

			// On timeout delete peer because its unresponsive
			fmt.Println("Timeout. Sending Block. Retrying...")
			delete(peerAddresses, peerAddress.String())
			// ensureRPC <- true
		}

	}else{

		delete(peerAddresses, peerAddress.String())
		// ensureRPC <- true
		outLog.Printf("Peer Unresponsive. Removed Peer:" + peerAddress.String())

	}	

}


// go-routine to process a block received and add to chain. 
// Move to next iteration when done

func addBlockToChain(block Block) {

	for block.Data.Iteration > iterationCount {
		outLog.Printf("Blocking. Got block for %d, I am at %d\n", block.Data.Iteration, iterationCount)
		time.Sleep(1000 * time.Millisecond)
	}

	outLog.Printf("Adding block to chain")	
	blockChainLock.Lock()
	err := client.addBlock(block)
	blockChainLock.Unlock()
	outLog.Printf("Adding block to chain")
	// TODO: check if this is required
	// boolLock.Lock()
	// boolLock Unlocked after lock in previous function

	if ((block.Data.Iteration == iterationCount) && (err ==nil)){
		outLog.Printf("Checking convergence")
		convergedLock.Lock()
		converged = client.checkConvergence()
		outLog.Printf("Convergence checked")
		convergedLock.Unlock()
		boolLock.Unlock()
		outLog.Printf("Bool lock released")				
		go sendBlock(block)	
	}else{
	
		boolLock.Unlock()
		outLog.Printf("Bool lock released")	
	
	}



}

// Main sending thread. Checks if you are a non-verifier in the current itearation 
// Sends update if thats the case.
// TODO: Replace with channels for cleanliness

func messageSender(ports []string) {

	for {

		if verifier {

			time.Sleep(100 * time.Millisecond)
			continue
		}

		outLog.Printf("Acquiring bool lock")
		boolLock.Lock()

		if !updateSent {

			outLog.Printf("Computing Update\n")

			client.computeUpdate(iterationCount, datasetName)

			portsToConnect = VRF(iterationCount)

			for _, port := range portsToConnect {
				
				sendUpdateToVerifier(port) // This has to be a go-routine. Can't keep the lock and wait for RPC to return. No matter what the result of the update RPC call. Lets dare to make it a non-go routine
				if iterationCount == client.update.Iteration {
					updateSent = true
				}
			}

			boolLock.Unlock()
			outLog.Printf("Bool lock released")


		} else {

			boolLock.Unlock()
			outLog.Printf("Bool lock released")
			time.Sleep(100 * time.Millisecond)

		}

	}

}

// Make RPC call to send update to verifier
// If you cant connect to verifier or verifier fails midway RPC, then append an empty block and move on
// Start timer for receiving registering block

func sendUpdateToVerifier(port string) {

	var ign bool
	c := make(chan error)

	conn, err := rpc.Dial("tcp", verifierIP+port)
	printError("Unable to connect to verifier", err)
	
	if(err == nil){
		
		defer conn.Close()
		outLog.Printf("Making RPC Call to Verifier. Sending Update, Iteration:%d\n", client.update.Iteration)
		go func() { c <- conn.Call("Peer.VerifyUpdate", client.update, &ign) }()
		select {
		case err := <-c:
			
			handleErrorFatal("Error in sending update", err)
			if(err!=nil){
				outLog.Printf("Update sent successfully")
			}
			go startBlockDeadlineTimer()

			// use err and result
		case <-time.After(timeoutRPC):

			// create Empty Block and Send
			outLog.Printf("Timeout. Sending Update. Retrying...")
			sendUpdateToVerifier(port)
			blockChainLock.Lock()
			blockToSend, err := client.createBlock(iterationCount)
			blockChainLock.Unlock()
			printError("Iteration: " + strconv.Itoa(iterationCount), err)
			if(err == nil){
				go sendBlock(*blockToSend)
			}
		}
	
	}else{

		outLog.Printf("Will try and create an empty block")
		blockChainLock.Lock()
		blockToSend, err := client.createBlock(iterationCount)
		blockChainLock.Unlock()		
		printError("Iteration: " + strconv.Itoa(iterationCount), err)
		if(err==nil){
			// outLog.Printf("T")
			outLog.Printf("Will try and create an empty block")
			go sendBlock(*blockToSend)
		}
		// create Empty Block and Send
	}
}

// Timer started by the verifier to set a deadline until which he will receive updates

func startUpdateDeadlineTimer(){

	outLog.Printf("Starting Update Deadline Timer. Iteration: %d", iterationCount)
	
	select{
		
		case <- allUpdatesReceived:
			outLog.Printf("All Updates Received. Preparing to send block..")

		case <-time.After(timeoutUpdate):
			outLog.Printf("Timeout. Didn't receive expected number of updates. Preparing to send block. Iteration: %d..", iterationCount)
	
	}

	
	if(len(client.blockUpdates) > 0){
		outLog.Printf("Acquiring chain lock")
		blockChainLock.Lock()
		outLog.Printf("chain lock acquired")
		blockToSend, err := client.createBlock(iterationCount)
		blockChainLock.Unlock()		
		printError("Iteration: " + strconv.Itoa(iterationCount), err)
		if(err==nil){
			sendBlock(*blockToSend)
		}		
	}else{
		outLog.Printf("Received no updates from peers. I WILL DIE")
		os.Exit(1)
	}

}

func startBlockDeadlineTimer(){

	
	select{
		
		case <- blockReceived:
			outLog.Printf("Block Received. Appending to chain and moving on to the next iteration..")

		case <-time.After(timeoutBlock):
			
			outLog.Printf("Timeout. Didn't receive block. Appending empty block. Iteration: ..")			
			blockChainLock.Lock()
			outLog.Printf("chain lock acquired")
			blockToSend, err := client.createBlock(iterationCount)
			blockChainLock.Unlock()		
			printError("Iteration: " + strconv.Itoa(iterationCount), err)
			if(err==nil){
				sendBlock(*blockToSend)
			}	
	}

}


