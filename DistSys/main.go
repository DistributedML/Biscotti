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

const (
	basePort     	int           = 8000
	myIP         	string        = "127.0.0.1:"
	verifierIP   	string        = "127.0.0.1:"
	timeoutRPC    	time.Duration = 10000000000
	numVerifiers 	int           = 1
	timeoutUpdate 	time.Duration = 10000000000  // will need experimenting to figure out best possible timeout
)

type Peer int

var (

	//Input arguments
	datasetName   		string
	numberOfNodes 		int
	myPort				string

	client 				Honest

	ensureRPC      		chan bool
	allUpdatesReceived	chan bool
	networkBootstrapped	chan bool
	portsToConnect 		[]string
	peerPorts   		[]string
	peerAddresses		map[string]net.TCPAddr

	//Locks
	updateLock    		sync.Mutex
	boolLock      		sync.Mutex
	convergedLock 		sync.Mutex
	peerLock			sync.Mutex
	blockChainLock		sync.Mutex

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

	for update.Iteration > iterationCount {
		outLog.Printf("Blocking. Got update for %d, I am at %d\n", update.Iteration, iterationCount)
		time.Sleep(1000 * time.Millisecond)
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

	for block.Data.Iteration > iterationCount {
		outLog.Printf("Blocking. Got block for %d, I am at %d\n", block.Data.Iteration, iterationCount)
		time.Sleep(1000 * time.Millisecond)
	}

	boolLock.Lock()
	
	if block.Data.Iteration != len(client.bc.blocks) - 1 {
		
		better := client.evaluateBlockQuality(block) // check equality and some measure of 	
		boolLock.Unlock()	

		if(better){
			
			// TODO: If I receive a better block than my current one. Then I replace my block with this one.
			// I request for all the next blocks. I will also need to advertise new block or not?
			// go callRequestChainRPC(same conn) // returns whole chain. Is it longer than mine?
			// go evaluateReceivedChain() // if chain longer than mine and checks out replace mine with his
			
			if(block.Data.Iteration == len(client.bc.blocks) - 2){
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

	boolLock.Unlock()

	go addBlockToChain(block)

	return nil

}

// Register a fellow node as a peer.
// Returns:
// 	-nil if peer added

func (s *Peer) RegisterPeer(peerAddress net.TCPAddr, _ignored *bool) error {

	outLog.Printf("Registering peer:" + peerAddress.String())
	peerLock.Lock()
	peerAddresses[peerAddress.String()] = peerAddress
	peerLock.Unlock()
	if(myPort == strconv.Itoa(basePort)){
		networkBootstrapped <- true
	}
	return nil
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

// Parse args, read dataset and initialize separate threads for listening for updates/blocks and sending updates

func main() {

	gob.Register(&net.TCPAddr{})
	
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

	ensureRPC = make(chan bool)
	allUpdatesReceived = make (chan bool)
	networkBootstrapped = make (chan bool)


	// Initializing RPC Server
	peer := new(Peer)
	peerServer := rpc.NewServer()
	peerServer.Register(peer)
	

	go messageListener(peerServer, myPort)

	// announce yourself to above calculated peers. The first node in the network doesn't need to do this. He waits for an incoming peer instead. 	
	if(myPort != strconv.Itoa(basePort)){
		announceToNetwork(potentialPeerList)
	}else{
		<- networkBootstrapped
	}

	prepareForNextIteration()
	
	messageSender(peerPorts)

}



// peers announce themselves to all other nodes when they come into the system 
// This helps them maintain a list of peers to which they can send blocks

// checking heartbeat of each peer periodically. If down, add to list of unresponsive peers.
// Use list of responsive peers when sending block.

//OR

// The only problem is the rpc dial for now. If rpc dial doesn't happen. Ignore and move on

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

	var ign bool
	c := make(chan error)

	conn, err := rpc.Dial("tcp", peerAddress.String()) 
	printError("Peer offline.Couldn't connect to peer: " + peerAddress.String(), err)
	

	if(err == nil){

		defer conn.Close()	
		outLog.Printf("Calling RPC:"+ peerAddress.String())
		go func() { c <- conn.Call("Peer.RegisterPeer", myAddress, &ign) }()
		outLog.Printf("RPC called"+ peerAddress.String())
		select {

		case err = <-c:

			handleErrorFatal("Error in registering peer", err)
			
			if(err == nil){
				outLog.Printf("Announced myself to a fellow peer at port")
				peerLock.Lock()
				peerAddresses[peerAddress.String()] = peerAddress
				peerLock.Unlock()
			}


			// use err and result
		case <-time.After(timeoutRPC):

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

	boolLock.Lock()

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
	for _, address := range peerAddresses {
		go callRegisterBlockRPC(block, address)
	}
	
	//check for convergence, wait for RPC calls to return and move to the new iteration



	ensureRPCCallsReturn()
	peerLock.Unlock()

	// You can only move to the next iteration by sending a block if you were the verifier for that iteration
	
	if(block.Data.Iteration == iterationCount && verifier){

		convergedLock.Lock()
		converged = client.checkConvergence()
		convergedLock.Unlock()

		prepareForNextIteration()

	}
		

}

// output from channel to ensure all RPC calls to broadcast block are successful

func ensureRPCCallsReturn() {

	for i := 0; i < len(peerAddresses); i++ {
		<-ensureRPC
	}

}

// RPC call to send block to one peer

func callRegisterBlockRPC(block Block, peerAddress net.TCPAddr) {

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
			ensureRPC <- true

			// use err and result
		case <-time.After(timeoutRPC):

			// On timeout delete peer because its unresponsive
			fmt.Println("Timeout. Sending Block. Retrying...")
			delete(peerAddresses, peerAddress.String())
			ensureRPC <- true
		}

	}else{

		delete(peerAddresses, peerAddress.String())
		ensureRPC <- true
		outLog.Printf("Peer Unresponsive. Removed Peer:" + peerAddress.String())

	}
	

}


// go-routine to process a block received and add to chain. 
// Move to next iteration when done

func addBlockToChain(block Block) {

	blockChainLock.Lock()
	client.bc.AddBlockMsg(block)
	blockChainLock.Unlock()

	// TODO: check if this is required
	if block.Data.Iteration == iterationCount {
		boolLock.Lock()
		updateSent = true
		boolLock.Unlock()
	}

	convergedLock.Lock()
	converged = client.checkConvergence()
	convergedLock.Unlock()
	go sendBlock(block)	
	prepareForNextIteration()

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

		boolLock.Lock()

		if !updateSent {

			outLog.Printf("Computing Update\n")

			client.computeUpdate(iterationCount, datasetName)

			portsToConnect = VRF(iterationCount)

			for _, port := range portsToConnect {
				go sendUpdateToVerifier(port)
				if iterationCount == client.update.Iteration {
					updateSent = true
				}
			}

			boolLock.Unlock()

		} else {

			boolLock.Unlock()
			time.Sleep(100 * time.Millisecond)

		}

	}

}

// Make RPC call to send update to verifier
// If you cant connect to verifier or verifier fails midway RPC, then append an empty block and move on
// TODO: Some clients are unable to connect to their verifier in the original system. They will propose an empty block. The general public should
// reject their block because they have better blocks from other verifiers

func sendUpdateToVerifier(port string) {

	var ign bool
	c := make(chan error)

	conn, err := rpc.Dial("tcp", verifierIP+port)
	printError("Unable to connect to verifier", err)
	outLog.Printf("Making RPC Call to Verifier. Sending Update, Iteration:%d\n", client.update.Iteration)

	if(err == nil){
		
		defer conn.Close()
		go func() { c <- conn.Call("Peer.VerifyUpdate", client.update, &ign) }()
		select {
		case err := <-c:
			
			handleErrorFatal("Error in sending update", err)
			if(err!=nil){
				outLog.Printf("Update sent successfully")
			}
			// use err and result
		case <-time.After(timeoutRPC):

			// create Empty Block and Send
			outLog.Printf("Timeout. Sending Update. Retrying...")
			sendUpdateToVerifier(port)
			blockToSend := client.createBlock(iterationCount)
			sendBlock(blockToSend)
		}
	
	}else{

		blockToSend := client.createBlock(iterationCount)
		sendBlock(blockToSend)
		// create Empty Block and Send
	}
	

}

func startUpdateDeadlineTimer(){

	

	select{
		
		case <- allUpdatesReceived:
			outLog.Printf("All Updates Received. Preparing to send block..")

		case <-time.After(timeoutUpdate):
			outLog.Printf("Timeout. Didn't receive expected number of updates. Preparing to send block..")
	
	}

	updateLock.Lock()
	if(len(client.blockUpdates) > 0){
		blockToSend := client.createBlock(iterationCount)
		updateLock.Unlock()
		sendBlock(blockToSend)
	}else{
		updateLock.Unlock()
		outLog.Printf("Received no updates from peers. I WILL DIE")
		os.Exit(1)
	}

}


