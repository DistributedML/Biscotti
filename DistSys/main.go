package main

import (
	
	"bufio"
    "errors"
	"fmt"
	"github.com/sbinet/go-python"
	"log"
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

//Assumption: Requires node 0 to be online first. Need to move away from this

const (
	basePort        int           = 8000
	verifierIP   	string        = "127.0.0.1:"
	timeoutRPC    	time.Duration = 10000000000
	timeoutUpdate 	time.Duration = 10000000000  
	timeoutBlock 	time.Duration = 15000000000  
	timeoutPeer 	time.Duration = 5000000000
	
	NUM_VERIFIERS 	int           = 1
	NUM_MINERS 		int           = 2
	DEFAULT_STAKE   int 		  = 10

	VERIFIER_PRIME 	int 		  = 2
	MINER_PRIME 	int 		  = 3

	PRECISION       int 		  = 4
	POLY_SIZE 		int 		  = 10
	TOTAL_SHARES 	int 		  = 10
	SECURE_AGG  	bool 		  = true

)

type Peer int

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
	myVRF				VRF
	
	allSharesReceived		chan bool
	allUpdatesReceived		chan bool
	networkBootstrapped		chan bool
	blockReceived 			chan bool
	portsToConnect 			[]string
	verifierPortsToConnect 	[]string
	minerPortsToConnect 	[]string
	peerPorts   			[]string
    peerLookup          	map[string]int
	peerAddresses			map[int]net.TCPAddr
	stakeMap				map[int]int
	// pkMap					map[int]PublicKey
	// commitKey 				PublicKey
	// sKey 					kyber.Scalar


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
	miner 				bool
	iterationCount 		= -1

	// these are maps since it optimizes contains()
	roleIDs				map[int]int

	//Logging
	errLog *log.Logger = log.New(os.Stderr, "[err] ", log.Lshortfile|log.LUTC|log.Lmicroseconds)
	outLog *log.Logger = log.New(os.Stderr, "[peer] ", log.Lshortfile|log.LUTC|log.Lmicroseconds)

	//Errors
	staleError error = errors.New("Stale Update/Block")
	roniError  error = errors.New("RONI Failed")
	rpcError  error = errors.New("RPC Timeout")

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

	outLog.Printf(strconv.Itoa(client.id)+":Got RONI message, iteration %d\n", update.Iteration)

	/*	// we can return the chain to the guy here instead of just leaving that guy with an error
	if update.Iteration < iterationCount {
		printError("Update of previous iteration received", staleError)
		// sender is stale, return true here and let them catch up
		return true
	}*/

	roniScore := client.verifyUpdate(update)
	outLog.Printf("RONI for update at iteration %d is %f.\n", client.update.Iteration, roniScore)

	// Roni score measures change in local training error
	if roniScore > 0.02 {
		outLog.Printf("Rejecting update!")		
		return roniError
	}

	// TODO: Instead of adding to a block, sign it and return to client
	return nil

}

// The peer receives an update from another peer if its a verifier in that round.
// The verifier peer takes in the update and returns immediately.
// It calls a separate go-routine for collecting updates and sending updates when all updates have been collected
// Returns:
// - StaleError if its an update for a preceding round.

func (s *Peer) RegisterSecret(share MinerPart, _ignored *bool) error {

	outLog.Printf(strconv.Itoa(client.id)+":Got miner request, iteration %d\n", share.Iteration)

	// we can return the chain to the guy here instead of just leaving that guy with an error
	if share.Iteration < iterationCount {
		printError("Share of previous iteration received", staleError)
		return staleError
	}

	// Process update only called by the miner nodes
	go processShare(share)

	return nil

}


// go routine to process the update received by miner nodes
func processShare(share MinerPart) {

	outLog.Printf(strconv.Itoa(client.id)+":Got update for %d, I am at %d\n", share.Iteration, iterationCount)

	for share.Iteration > iterationCount {
		outLog.Printf(strconv.Itoa(client.id)+":Blocking for stale update. Update for %d, I am at %d\n", share.Iteration, iterationCount)
		time.Sleep(2000 * time.Millisecond)
	}

	// Might get an update while I am in the announceToNetwork phase and when I come out of it the update becomes redundant
	if ((iterationCount == share.Iteration)) {

		updateLock.Lock()
		numberOfShares := client.addSecretShare(share)
		updateLock.Unlock()

		//send signal to start sending Block if all updates Received. Changed this from numVanilla stuff
		if numberOfShares == numberOfNodeUpdates {			
			outLog.Printf(strconv.Itoa(client.id)+":All shares for iteration %d received. Notifying channel.", iterationCount)	
			allSharesReceived <- true 		 
		}
		
	}

}

// The peer receives an update from another peer if its a verifier in that round.
// The verifier peer takes in the update and returns immediately.
// It calls a separate go-routine for collecting updates and sending updates when all updates have been collected
// Returns:
// - StaleError if its an update for a preceding round.

func (s *Peer) RegisterUpdate(update Update, _ignored *bool) error {

	outLog.Printf(strconv.Itoa(client.id)+":Got miner request, iteration %d\n", update.Iteration)

	// we can return the chain to the guy here instead of just leaving that guy with an error
	if update.Iteration < iterationCount {
		printError("Update of previous iteration received", staleError)
		return staleError
	}

	// Process update only called by the miner nodes
	go processUpdate(update)

	return nil

}

// The peer receives a block from the verifier of that round.
// It takes in the block and returns immediately.
// It calls a separate go-routine for appending the block as part of its chain
// Returns:
// - staleError if its an block for a preceding round.

func (s *Peer) RegisterBlock(block Block, returnBlock *Block) error {

	outLog.Printf(strconv.Itoa(client.id)+":Got block message, iteration %d\n", block.Data.Iteration)

	// can return the latest block I have but there is no need yet
	*returnBlock = block

	go processBlock(block)

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

	outLog.Printf(strconv.Itoa(client.id) + ":Registering peer:" + peerAddress.String())
	peerLock.Lock()
	peerAddresses[peerLookup[peerAddress.String()]] = peerAddress
	stakeMap[peerLookup[peerAddress.String()]] = DEFAULT_STAKE
	peerLock.Unlock()
	// if I am first node (index:0) and I am waiting for a peer to join (iterationCount < 0) then send signal that I have atleast one peer.
	if(myPort == strconv.Itoa(basePort) && iterationCount < 0){
		networkBootstrapped <- true
	}
	*chain = *client.bc

	outLog.Printf("Advertising chain to peer with length %d", len(chain.Blocks))

	return  nil 
}

func (s *Peer) GetUpdateList(iteration int, nodeList *([]int)) error {

	outLog.Printf(strconv.Itoa(client.id) + ":Giving update list")
	
	if(iteration == iterationCount){

		// might need to & it
		for nodeID, _ := range client.secretList {
			*(nodeList) = append((*nodeList), nodeID)
		}
	
	}else{
		
		return staleError
	
	}

	return nil

}

func (s *Peer) GetMinerPart(nodeList []int, myMinerPart *MinerPart) error {

	outLog.Printf(strconv.Itoa(client.id) + ":Giving my aggregated share")
	
	*(myMinerPart) = MinerPart{}
	// Might need this condition
	if(miner && (client.secretList[nodeList[0]].Iteration == iterationCount)){

		*(myMinerPart) = client.secretList[nodeList[0]]

		for i := 1; i < len(nodeList); i++ {
			
			*(myMinerPart) = aggregateSecret(*(myMinerPart), client.secretList[nodeList[i]])
		}

	
	}else{
		
		return staleError
	
	}

	return nil

}
// Basic check to see if you are the verifier in the next round
func amVerifier(nodeNum int) bool {
	return roleIDs[client.id] % VERIFIER_PRIME == 0
}

// Basic check to see if you are the verifier in the next round
func amMiner(nodeNum int) bool {
	return roleIDs[client.id] % MINER_PRIME == 0
}

// Runs a single VRF to get roleIDs. Don't rerun this.
func getRoles() map[int]int {
	
	roleMap := make(map[int]int)
	for i := 0; i < numberOfNodes; i++ {
		roleMap[i] = 1
	}

	vIDs, mIDs, noisers, _, _ := myVRF.getNodes(stakeMap, client.bc.getLatestBlockHash(), 
		NUM_VERIFIERS, NUM_MINERS, numberOfNodes)

	outLog.Printf("Verifiers are %s", vIDs)
	outLog.Printf("Miners are %s", mIDs)
	outLog.Printf("Noisers are %s", noisers)

	for _, id := range vIDs {
		roleMap[id] *= VERIFIER_PRIME
	}

	for _, id := range mIDs {
		roleMap[id] *= MINER_PRIME
	}

	return roleMap
}

// Convert the roleIDs to verifier/miner strings 
func getRoleNames(iterationCount int) ([]string, []string, int) {

	verifiers := make([]string, 0)
	miners := make([]string, 0)
	numVanilla := 0

    // Find the address corresponding to the ID.
    // TODO: Make fault tolerant
    // TODO: Maybe implement inverted index
    for address, ID := range peerLookup {

    	if (roleIDs[ID] == 1) {
    		numVanilla++
    	}
        
    	if (roleIDs[ID] % VERIFIER_PRIME) == 0 {
    		verifiers = append(verifiers, address)
    	}

    	if (roleIDs[ID] % MINER_PRIME) == 0 {
    		miners = append(miners, address)
    	}

    }

	return verifiers, miners, numVanilla

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

// // TODO:
// func extractPublicKeys() map[int]PublicKey{

// }

// // TODO:
// func extractSecretKey(int nodeNum) map[int]PublicKey{
	
// }

// // TODO:
// func extractCommitmentKey(){

// }



// Parse args, read dataset and initialize separate threads for listening for updates/Blocks and sending updates

func main() {

	gob.Register(&net.TCPAddr{})
	gob.Register(&Blockchain{})
	
	//Parsing arguments nodeIndex, numberOfNodes, datasetname
	numberOfNodesPtr := flag.Int("t", 0 , "The total number of nodes in the network")

	nodeNumPtr := flag.Int("i", -1 ,"The node's index in the total. Has to be greater than 0")

	datasetNamePtr := flag.String("d", "" , "The name of the dataset to be used")

    peersFileNamePtr := flag.String("f", "", "File that contains list of IP:port pairs")

    myIPPtr := flag.String("a", "", " If not local, this node's IP")

    myPrivateIPPtr := flag.String("pa", "", " If not local, this node's private IP")

    myPortPtr := flag.String("p", "", " If not local, this node's port")

	flag.Parse()

	nodeNum := *nodeNumPtr
	numberOfNodes = *numberOfNodesPtr
	datasetName = *datasetNamePtr
    datasetName = *datasetNamePtr
    peersFileName = *peersFileNamePtr
    myPrivateIP = *myPrivateIPPtr+":"
    myIP = *myIPPtr+":"
    myPort = *myPortPtr

	if(numberOfNodes <= 0 || nodeNum < 0 || datasetName == ""){
		flag.PrintDefaults()
		os.Exit(1)	
	}

    // getports of all other clients in the system
    peerLookup = make(map[string]int)
    
    // Initialize default; uniform stake
    stakeMap = make(map[int]int)
        
    potentialPeerList := make([]net.TCPAddr, 0, numberOfNodes-1)

    // Running locally
    if (peersFileName == "") {

        myIP = "127.0.0.1:"
        myPort = strconv.Itoa(nodeNum + basePort)

        for i := 0; i < numberOfNodes; i++ {
                
            peerPort := strconv.Itoa(basePort+i)

            if peerPort == myPort {
                peerLookup[fmt.Sprintf(myIP + peerPort)] = i
                stakeMap[i] = DEFAULT_STAKE
                continue
            }
            
            peerPorts = append(peerPorts, peerPort)
            peerAddress, err := net.ResolveTCPAddr("tcp", fmt.Sprintf(myIP + peerPort))
            handleErrorFatal("Unable to resolve a potential peer address", err)
            potentialPeerList = append(potentialPeerList, *peerAddress)
            peerLookup[fmt.Sprintf(myIP + peerPort)] = i
            stakeMap[i] = DEFAULT_STAKE
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
                stakeMap[i] = DEFAULT_STAKE
                continue
            }

            peerAddress, err := net.ResolveTCPAddr("tcp", peerAddressStr)
            handleErrorFatal("Unable to resolve a potential peer address", err)
            potentialPeerList = append(potentialPeerList, *peerAddress)
            peerLookup[peerAddressStr] = i
            stakeMap[i] = DEFAULT_STAKE
        }

        if !nodeInList {
            handleErrorFatal("Node is not in peer list", errors.New(""))
        }

    }

    // init peer addresses list
    peerAddresses = make(map[int]net.TCPAddr)

    // pkMap := extractPublicKeys()
    // commitKey := extractCommitmentKey()
    // sKey := extractSecretkey(nodeNum)


	//Initialize a honest client
	client = Honest{id: nodeNum, blockUpdates: make([]Update, 0, 5)}

	
	// Reading data and declaring some global locks to be used later
	client.initializeData(datasetName, numberOfNodes)
	client.bootstrapKeys()

	// initialize the VRF
	myVRF = VRF{}
	myVRF.init()

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
	allSharesReceived = make (chan bool)


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
	messageSender(peerPorts)
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
	chain := Blockchain{[]*Block{}}
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
		go func() { c <- conn.Call("Peer.RegisterPeer", myAddress, &chain) }()
		outLog.Printf(strconv.Itoa(client.id)+":RPC called"+ peerAddress.String())
		select {

		case err = <-c:

			if(err == nil){
				
				outLog.Printf(strconv.Itoa(client.id)+":Announced myself to a fellow peer at port. Got latest chain")
				
				//Add peer
				peerLock.Lock()
				peerAddresses[peerLookup[peerAddress.String()]] = peerAddress
				peerLock.Unlock()

				outLog.Printf("Chain length received:%d", len(chain.Blocks))
				outLog.Printf("My chain length:%d", len(client.bc.Blocks))

				//Check the chain and see if its the longest one. If longer replace it with mine
				if(len(chain.Blocks) > len(client.bc.Blocks)){
					boolLock.Lock()
					iterationCount = client.replaceChain(chain)
					if(len(client.bc.Blocks) - 2 != iterationCount) {
						outLog.Printf("Iteration:%d", iterationCount)
						outLog.Printf("Chain Length:%d", len(client.bc.Blocks))	

						outLog.Printf("Chain and iteration Count are inconsistent",)
						os.Exit(1)
					}
					outLog.Printf(strconv.Itoa(client.id)+"Got lastest chain until iteration " + strconv.Itoa(iterationCount))
					boolLock.Unlock()
				}

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
		client.bc.PrintChain()
		os.Exit(1)
	}

	if(SECURE_AGG) {

		updateLock.Lock()
		client.flushSecrets()
		updateLock.Unlock()
	
	}else{

		updateLock.Lock()
		client.flushUpdates()
		updateLock.Unlock()	
	}	

	convergedLock.Unlock()
	boolLock.Lock()

	
	iterationCount++
	outLog.Printf("Moving on to next iteration %d", iterationCount)

	// This runs the VRF and sets the verifiers for this iteration
	roleIDs = getRoles()
	verifier = amVerifier(client.id)
	miner = amMiner(client.id)

	if miner {
		outLog.Printf(strconv.Itoa(client.id)+":I am miner. Iteration:%d", iterationCount)
		updateSent = true
		if (SECURE_AGG) {
			go startShareDeadlineTimer(iterationCount)
		}else{
			go startUpdateDeadlineTimer(iterationCount) //start timer for receiving updates
		}

	} else if verifier {

		outLog.Printf(strconv.Itoa(client.id)+":I am verifier. Iteration:%d", iterationCount)
		updateSent = true
	
	} else {
		outLog.Printf(strconv.Itoa(client.id)+":I am not miner or verifier. Iteration:%d", iterationCount)
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

		//send signal to start sending Block if all updates Received. Changed this from numVanilla stuff
		if numberOfUpdates == (numberOfNodeUpdates) {			
			outLog.Printf(strconv.Itoa(client.id)+":All updates for iteration %d received. Notifying channel.", iterationCount)	
			allUpdatesReceived <- true 		 
		}	
	
	}

}



// func processBlock(block Block) {
	
// 	// Lock to ensure that iteration count doesn't change until I have appended block
// 	outLog.Printf("Trying to acquire lock...")
// 	boolLock.Lock()

// 	outLog.Printf("Got lock, processing block")

// 	hasBlock := client.hasBlock(block.Data.Iteration)

// 	// Block is old, but could be better than my current block
// 	if ((block.Data.Iteration < iterationCount) || hasBlock || iterationCount<0) {
				
// 		boolLock.Unlock()

// 		if hasBlock {
// 			outLog.Printf("Already have block")
// 		}

// 		if (iterationCount < 0) {
// 			return
// 		}

// 		better := client.evaluateBlockQuality(block) // check equality and some measure of 	

// 		if better {
			
// 			// TODO: If I receive a better block than my current one. Then I replace my block with this one.
// 			// I request for all the next Blocks. I will also need to advertise new block or not?
// 			// go callRequestChainRPC(same conn) // returns whole chain. Is it longer than mine?
// 			// go evaluateReceivedChain() // if chain longer than mine and checks out replace mine with his
			
// 			outLog.Printf("Chain Length:" + strconv.Itoa(len(client.bc.Blocks)))
// 			if(block.Data.Iteration == len(client.bc.Blocks) - 2){
// 				client.replaceBlock(block, block.Data.Iteration)
// 				outLog.Printf("Chain Length:" + strconv.Itoa(len(client.bc.Blocks)))
// 				outLog.Printf(strconv.Itoa(client.id)+":Received better block")
// 				return 
// 			}

		
// 		} else {
			
// 			// returnBlock = client.bc.getBlock(block.Data.Iteration)						
// 			outLog.Printf(strconv.Itoa(client.id)+":Equal block")
// 			return 
		
// 		}

	 
// 	}

// 	if block.Data.Iteration > iterationCount {
		
// 		boolLock.Unlock()

// 		for block.Data.Iteration > iterationCount {
// 			outLog.Printf(strconv.Itoa(client.id)+":Blocking. Got block for %d, I am at %d\n", block.Data.Iteration, iterationCount)
// 			time.Sleep(1000 * time.Millisecond)
// 		}

// 		boolLock.Lock()
// 	}
	
// 	go addBlockToChain(block)


// }

// // For all non-miners, accept the block

func processBlock(block Block) {

	// Lock to ensure that iteration count doesn't change until I have appended block

	outLog.Printf("Trying to acquire lock...")
	boolLock.Lock()

	outLog.Printf("Got lock, processing block")

	hasBlock := client.hasBlock(block.Data.Iteration)

	if ((block.Data.Iteration < iterationCount) || hasBlock || iterationCount<0) {
		
		if hasBlock {
			outLog.Printf("Already have block")
		}
		
		boolLock.Unlock()
		outLog.Printf("Released bool lock")

		if(iterationCount< 0){
			return
		}

		better := client.evaluateBlockQuality(block) // check equality and some measure of 	

		if(better){
			
			// TODO: If I receive a better block than my current one. Then I replace my block with this one.
			// I request for all the next Blocks. I will also need to advertise new block or not?
			// go callRequestChainRPC(same conn) // returns whole chain. Is it longer than mine?
			// go evaluateReceivedChain() // if chain longer than mine and checks out replace mine with his
			
			outLog.Printf("Chain Length:" + strconv.Itoa(len(client.bc.Blocks)))
			if(block.Data.Iteration == len(client.bc.Blocks) - 2){
				client.replaceBlock(block, block.Data.Iteration)
				outLog.Printf("Chain Length:" + strconv.Itoa(len(client.bc.Blocks)))
				outLog.Printf(strconv.Itoa(client.id)+":Received better  block")
			}
			return 


		
		}else{
			
			// returnBlock = client.bc.getBlock(block.Data.Iteration)						
			outLog.Printf(strconv.Itoa(client.id)+":Equal block")
			return 
		
		}

		// handleErrorFatal("Block of previous iteration received", staleError)
	}

	if block.Data.Iteration > iterationCount {
		
		boolLock.Unlock()
		outLog.Printf("Released bool lock")

		for block.Data.Iteration > iterationCount {
			outLog.Printf(strconv.Itoa(client.id)+":Blocking. Got block for %d, I am at %d\n", block.Data.Iteration, iterationCount)
			time.Sleep(1000 * time.Millisecond)
		}

		

		if ((block.Data.Iteration == iterationCount) || client.evaluateBlockQuality(block)){

			outLog.Printf("Acquiring bool lock")
			boolLock.Lock()	

			go addBlockToChain(block)

		}

		return
	}
			
	// // if not empty and not verifier send signal to channel. Not verifier required because you are not waiting for a block if you are the verifier and if you receive an empty block and if you are currently busy bootstrapping yourself. 
	// if(len(block.Data.Deltas) != 0 && !verifier && iterationCount >= 0) {
	// 	blockReceived <- true
	// }

	
	go addBlockToChain(block)

}

// go-routine to process a block received and add to chain. 
// Move to next iteration when done

func addBlockToChain(block Block) {

	// Add the block to chain
	blockChainLock.Lock()
	
	outLog.Printf(strconv.Itoa(client.id)+":Adding block for %d, I am at %d\n", 
		block.Data.Iteration, iterationCount)
	
	err := client.addBlock(block)
	blockChainLock.Unlock()

	if ((block.Data.Iteration == iterationCount) && (err == nil)){
	
		// If block is current, notify channel waiting for it
		if(len(block.Data.Deltas) != 0 && updateSent && !verifier && iterationCount >= 0) {
			outLog.Printf(strconv.Itoa(client.id)+":Sending block to channel")
			blockReceived <- true
		
		}

		boolLock.Unlock()
		go sendBlock(block)	
	
	}else{
	
		boolLock.Unlock()
		outLog.Printf(strconv.Itoa(client.id)+":Bool lock released")	
	
	}

}


// Miner broadcasts the block of this iteration to all peers
func sendBlock(block Block) {	

	outLog.Printf(strconv.Itoa(client.id)+":Sending block of iteration: %d\n", block.Data.Iteration)

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

	outLog.Printf(strconv.Itoa(client.id)+":RPC calls successfully returned. Iteration: %d", iterationCount)

	convergedLock.Lock()
	converged = client.checkConvergence()
	convergedLock.Unlock()

	outLog.Printf(strconv.Itoa(client.id)+":Preparing for next Iteration. Current Iteration: %d", iterationCount)

	prepareForNextIteration()
		
}

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

			outLog.Printf(strconv.Itoa(client.id)+":Block sent to peer successful. Peer: " + peerAddress.String() + " Iteration: %d", block.Data.Iteration)
			printError("Error in sending block", err)
			// ensureRPC <- true

			// use err and result
		case <-time.After(timeoutRPC):

			// On timeout delete peer because its unresponsive
			outLog.Printf("Timeout. Sending Block.")
			delete(peerAddresses, peerLookup[peerAddress.String()])
            // delete(peerLookup, peerAddress.String())
			// ensureRPC <- true
		}

	}else{

		delete(peerAddresses, peerLookup[peerAddress.String()])
        
		//BUG ALERT: You can't delete elements from peerLookup. VRF hinges on it. Its not going to be able to return the address in the verifier set
        // delete(peerLookup, peerAddress.String())

		// ensureRPC <- true
		outLog.Printf(strconv.Itoa(client.id)+":Peer Unresponsive. Removed Peer:" + peerAddress.String())

	}	

}

// Main sending thread. Checks if you are a non-verifier in the current itearation 
// Sends update if thats the case.

func messageSender(ports []string) {

	for {

		if verifier || miner {
			time.Sleep(100 * time.Millisecond)
			continue
		}

		boolLock.Lock()

		if !updateSent {

			outLog.Printf(strconv.Itoa(client.id)+":Computing Update\n")

			client.computeUpdate(iterationCount, datasetName)

			verifierPortsToConnect, minerPortsToConnect, 
				numberOfNodeUpdates = getRoleNames(iterationCount)
			
			outLog.Printf("Sending update to verifiers")
			approved := sendUpdateToVerifiers(verifierPortsToConnect)			 

			if approved {
				
				// break update down into secrets
				// send secrets to miners

				outLog.Printf("Sending update to miners")
				sendUpdateSecretsToMiners(minerPortsToConnect)
				
				sendUpdateToMiners(minerPortsToConnect)
			
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
// Start timer for receiving registering block

func sendUpdateToVerifiers(addresses []string) bool {

	var ign bool
	c := make(chan error)
	verified := false

	for _, address := range addresses {

		conn, err := rpc.Dial("tcp", address)
		printError("Unable to connect to verifier", err)
		
		if(err == nil){
			
			defer conn.Close()
			outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Verifier. Sending Update, Iteration:%d\n", client.update.Iteration)
			go func() { c <- conn.Call("Peer.VerifyUpdate", client.update, &ign) }()
			select {
			case verifierError := <-c:
				
				printError("Error in sending update", err)
				if (verifierError == nil) {
					outLog.Printf(strconv.Itoa(client.id)+":Update verified. Iteration:%d\n", client.update.Iteration)
					verified = true
				}

			// use err and result
			case <-time.After(timeoutRPC):
				outLog.Printf(strconv.Itoa(client.id)+":RPC Call timed out.")
				continue
			}
		
		} else {

			outLog.Printf("GOT VERIFIER ERROR")
			time.Sleep(1000 * time.Millisecond)

			continue
		}
	
	}

	// Verification totally failed. Create empty block and send
	if !verified {
		outLog.Printf(strconv.Itoa(client.id)+":Will try and create an empty block")
		blockChainLock.Lock()
		blockToSend, err := client.createBlock(iterationCount)
		blockChainLock.Unlock()		
		printError("Iteration: " + strconv.Itoa(iterationCount), err)
		if(err==nil){
			outLog.Printf(strconv.Itoa(client.id)+":Sending an empty block")
			go sendBlock(*blockToSend)
		}
	} 

	return verified

}

func sendUpdateToMiners(addresses []string) {

	var ign bool
	c := make(chan error)

	mined := false

	// TODO: For now, the first miner that gets the block done is good enough.
	// We will need to use shamir secrets here later
	for _, address := range addresses {

		if !mined {

			conn, err := rpc.Dial("tcp", address)
			printError("Unable to connect to miner", err)
			
			if (err == nil) {
				
				defer conn.Close()
				outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Miner. Sending Update, Iteration:%d\n", client.update.Iteration)
				go func() { c <- conn.Call("Peer.RegisterUpdate", client.update, &ign) }()
				select {
				case err := <-c:
					
					printError("Error in sending update", err)
					if(err==nil){
						outLog.Printf(strconv.Itoa(client.id)+":Update mined. Iteration:%d\n", client.update.Iteration)
						mined = true
					}

					if(err==staleError){
						outLog.Printf(strconv.Itoa(client.id)+"Stale error:Update mined. Iteration:%d\n", client.update.Iteration)
						mined = true
					}
					
					go startBlockDeadlineTimer(iterationCount)

					// use err and result
				case <-time.After(timeoutRPC):
					outLog.Printf(strconv.Itoa(client.id)+":RPC Call timed out.")
					continue
				}
			
			} else {
				
				outLog.Printf("GOT MINER ERROR")
				time.Sleep(1000 * time.Millisecond)

				continue
			}

		}

	}

	// Couldn't mine the block. Send empty block. // Why do we need this?
	if !mined {
		outLog.Printf(strconv.Itoa(client.id)+":Will try and create an empty block")
		blockChainLock.Lock()
		blockToSend, err := client.createBlock(iterationCount)
		blockChainLock.Unlock()		
		printError("Iteration: " + strconv.Itoa(iterationCount), err)
		if(err==nil){
			outLog.Printf(strconv.Itoa(client.id)+":Sending an empty block")
			go sendBlock(*blockToSend)
		}
	}

}

func sendUpdateSecretsToMiners(addresses []string) {

	var ign bool
	c := make(chan error)

	mined := false

	// generate secrets here
	minerSecrets := generateMinerSecretShares(client.update.Delta, PRECISION, client.Keys.CommitmentKey, NUM_MINERS, POLY_SIZE, TOTAL_SHARES)

	// fmt.Println(minerSecrets)
	// fmt.Println(minerSecrets[0].PolyMap[10].Secrets)
	// fmt.Println(minerSecrets[1].PolyMap[10].Secrets)
	// os.Exit(1)


	// // TODO: For now, the first miner that gets the block done is good enough.
	// // We will need to use shamir secrets here later

	minerIndex := 0

	for _, address := range addresses {

		if !mined {

			conn, err := rpc.Dial("tcp", address)
			printError("Unable to connect to miner", err)
			
			if (err == nil) {
				
				minerSecrets[minerIndex].Iteration = client.update.Iteration
				minerSecrets[minerIndex].NodeID = client.id
				
				defer conn.Close()
				outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Miner. Sending Update Share, Iteration:%d\n", client.update.Iteration)
				go func() { c <- conn.Call("Peer.RegisterSecret", minerSecrets[minerIndex], &ign) }()
				select {
				case err := <-c:
					
					printError("Error in sending secret share", err)
					if(err==nil){
						outLog.Printf(strconv.Itoa(client.id)+":Secret shared. Iteration:%d\n", client.update.Iteration)
						mined = true
					}

					if(err==staleError){
						outLog.Printf(strconv.Itoa(client.id)+"Stale error:Secret shared. Iteration:%d\n", client.update.Iteration)
						mined = true
					}
					
					go startBlockDeadlineTimer(iterationCount)

					// use err and result
				case <-time.After(timeoutRPC):
					outLog.Printf(strconv.Itoa(client.id)+":RPC Call timed out.")
					continue
				}
			
			} else { 
				
				outLog.Printf("GOT MINER ERROR. Unable to share secret")
				time.Sleep(1000 * time.Millisecond)

				continue
			}

		}

		minerIndex++

	}

	// Couldn't mine the block. Send empty block. // Why do we need this?
	if !mined {
		outLog.Printf(strconv.Itoa(client.id)+":Will try and create an empty block")
		blockChainLock.Lock()
		blockToSend, err := client.createBlock(iterationCount)
		blockChainLock.Unlock()		
		printError("Iteration: " + strconv.Itoa(iterationCount), err)
		if(err==nil){
			outLog.Printf(strconv.Itoa(client.id)+":Sending an empty block")
			go sendBlock(*blockToSend)
		}
	}

}

// Timer started by the verifier to set a deadline until which he will receive updates

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
	
			outLog.Printf(strconv.Itoa(client.id)+":Acquiring chain lock")
			blockChainLock.Lock()
			
			outLog.Printf(strconv.Itoa(client.id)+":chain lock acquired")
			blockToSend, err := client.createBlock(iterationCount)
			
			blockChainLock.Unlock()		
			printError("Iteration: " + strconv.Itoa(iterationCount), err)
			
			if (err == nil) {
				sendBlock(*blockToSend)
			}

		} else {

			outLog.Printf(strconv.Itoa(client.id)+":Received no updates from peers. I WILL DIE")
			os.Exit(1)
		}

	// An old timer was triggered, try to catch up. HOW DOES THIS HELP YOU CATCH UP
	} else {
		time.Sleep(1000 * time.Millisecond)
		outLog.Printf(strconv.Itoa(client.id)+":Forwarding timer ahead.")
		allUpdatesReceived <- true
	}

}

// Timer started by the verifier to set a deadline until which he will receive updates

func startShareDeadlineTimer(timerForIteration int){
	
	select {
		
		case <- allSharesReceived:
			outLog.Printf(strconv.Itoa(client.id)+":All Shares Received for timer on %d. I am at %d. Preparing to send block..", 
				timerForIteration, iterationCount)

		case <- time.After(timeoutUpdate):
			outLog.Printf(strconv.Itoa(client.id)+":Timeout. Didn't receive expected number of shares. Preparing to send block. Iteration: %d..", iterationCount)
	
	}
	
	// If I am on the current iteration and am the CHOSEN ONE among the miners

	if (timerForIteration == iterationCount)   {

		if ((myIP+myPort) == minerPortsToConnect[0]) {

			minerMap, finalNodeList := getNodesList(minerPortsToConnect)

			sharesPerMiner := TOTAL_SHARES/NUM_MINERS

			if (sharesPerMiner * len(minerMap) >= maxPolynomialdegree ) {

				client.aggregatedSecrets = getSecretShares(minerMap, finalNodeList)

				if (sharesPerMiner * len(minerMap) >= maxPolynomialdegree ) {

					outLog.Printf(strconv.Itoa(client.id)+":Acquiring chain lock")
					blockChainLock.Lock()
				
					outLog.Printf(strconv.Itoa(client.id)+":chain lock acquired")
					
					// //TODO:
					blockToSend, err := client.createBlockSecAgg(iterationCount, finalNodeList)
				
					blockChainLock.Unlock()		

					printError("Iteration: " + strconv.Itoa(iterationCount), err)
				
					if (err == nil) {
						sendBlock(*blockToSend)
					}					

				}else{
					outLog.Printf(strconv.Itoa(client.id)+":Creating empty block")
					dummyNodeList := make([]int,0)
					//create empty block
					blockChainLock.Lock()				
					outLog.Printf(strconv.Itoa(client.id)+":chain lock acquired")					
					// //TODO:
					blockToSend, err := client.createBlockSecAgg(iterationCount, dummyNodeList)				
					blockChainLock.Unlock()	

					if (err == nil) {
						sendBlock(*blockToSend)
					}				
				}

				
			}else{
					outLog.Printf(strconv.Itoa(client.id)+":Creating empty block")
					// create empty block
					outLog.Printf(strconv.Itoa(client.id)+":Creating empty block")
					dummyNodeList := make([]int,0)
					//create empty block
					blockChainLock.Lock()				
					outLog.Printf(strconv.Itoa(client.id)+":chain lock acquired")					
					// //TODO:
					blockToSend, err := client.createBlockSecAgg(iterationCount, dummyNodeList)				
					blockChainLock.Unlock()	
					if (err == nil) {
						sendBlock(*blockToSend)
					}
			}
		

		}else{

			go startBlockDeadlineTimer(iterationCount)

		} 

	// An old timer was triggered, try to catch up
	} else {

		time.Sleep(1000 * time.Millisecond)
		outLog.Printf(strconv.Itoa(client.id)+":Forwarding timer ahead.")
		allSharesReceived <- true
	
	}

}

func getSecretShares(minerList map[string][]int, nodeList []int) []MinerPart{

	minerShares := make([]MinerPart, 0)

	// aggreegating and appending my own list

	myMinerPart := client.secretList[nodeList[0]]

	for i := 1; i < len(nodeList); i++ {
		
		myMinerPart = aggregateSecret(myMinerPart, client.secretList[nodeList[i]])

	}

	minerShares = append(minerShares, myMinerPart)	

	for address, _ := range minerList {
		
		outLog.Printf(strconv.Itoa(client.id)+":Calling %s", address)
		thisMinerSecret, err := callGetMinerShareRPC(address, nodeList)		    
		
		if err == nil{
			minerShares = append(minerShares, thisMinerSecret)
		}

	}

	return minerShares
	
}


func callGetMinerShareRPC(address string, nodeList []int) (MinerPart, error){

	thisMinerPart := MinerPart{}
	
	c := make(chan error)

	// TODO: For now, the first miner that gets the block done is good enough.
	// We will need to use shamir secrets here later

	conn, err := rpc.Dial("tcp", address)
	printError("Unable to connect to fellow miner", err)
	
	if (err == nil) {
		
		defer conn.Close()
		outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Fellow Miner. Getting NodeList, Iteration:%d\n", client.update.Iteration)
		go func() { c <- conn.Call("Peer.GetMinerPart", nodeList, &thisMinerPart) }()
		select {
		case err := <-c:
			
			printError("Error in getting miner part", err)
			
			if(err==nil){
				outLog.Printf(strconv.Itoa(client.id)+":Update mined. Iteration:%d\n", client.update.Iteration)
				return thisMinerPart, nil
			}else{
				return thisMinerPart, err
			}		

			// use err and result
		case <-time.After(timeoutRPC):
			outLog.Printf(strconv.Itoa(client.id)+":RPC Call timed out.")
			return thisMinerPart, rpcError
		}
	
	} else {
		
		outLog.Printf("Unable to connect to fellow miner")
		time.Sleep(1000 * time.Millisecond)
		return thisMinerPart, err
	}

}

func getNodesList(minerList []string) (map[string][]int, []int) {
	
	listOfUpdates := make(map[string][]int, 0)

	myNodeList := make([]int, 0, len(client.secretList))

	for nodeID := range myNodeList {
		listOfUpdates[myIP+myPort] = append(listOfUpdates[myIP+myPort], nodeID)
	}


	// popuate list of updates with node list of each online and synchronous miner

	for _, address := range minerList {
        
        if (address == myIP+myPort){
        	continue
        }

        outLog.Printf(strconv.Itoa(client.id)+":Calling %s", address)
		thisNodesList, err := callGetUpdateListRPC(address)
		if err == nil {
			listOfUpdates[address] = thisNodesList	
			
		}

	}

	// find node ids whose updates are available with all online miners.

	intersectionList := listOfUpdates[myIP+myPort]

	for _, nodeList := range listOfUpdates {

		intersectionList = Intersection(intersectionList, nodeList)

	}

	return listOfUpdates, intersectionList

}


func callGetUpdateListRPC(address string) ([]int, error){

	nodeList := []int{}
	
	c := make(chan error)

	// TODO: For now, the first miner that gets the block done is good enough.
	// We will need to use shamir secrets here later
	

	conn, err := rpc.Dial("tcp", address)
	printError("Unable to connect to fellow miner", err)
	
	if (err == nil) {
		
		defer conn.Close()
		outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Fellow Miner. Getting NodeList, Iteration:%d\n", client.update.Iteration)
		go func() { c <- conn.Call("Peer.GetUpdateList", client.update.Iteration, &nodeList) }()
		select {
		case err := <-c:
			
			printError("Error in sending update", err)
			
			if(err==nil){
				outLog.Printf(strconv.Itoa(client.id)+":Update mined. Iteration:%d\n", client.update.Iteration)
				return nodeList, nil
			}else{
				return nodeList, err
			}		

			// use err and result
		case <-time.After(timeoutRPC):
			outLog.Printf(strconv.Itoa(client.id)+":RPC Call timed out.")
			return nodeList, rpcError
		}
	
	} else {
		
		outLog.Printf("Unable to connect to fellow miner")
		time.Sleep(1000 * time.Millisecond)
		return nodeList, err


	}

}
	
func startBlockDeadlineTimer(timerForIteration int){
	
	select{
		
		case <- blockReceived:
			
			outLog.Printf(strconv.Itoa(client.id)+":Channel for block at iteration: %d", timerForIteration)

			if (timerForIteration == iterationCount) {
				outLog.Printf(strconv.Itoa(client.id)+":Block received at current iteration. Appending to chain and moving on to the next iteration. %d", iterationCount)
			}

		case <-time.After(timeoutBlock):
		
			if (timerForIteration == iterationCount) {

				outLog.Printf(strconv.Itoa(client.id)+":Timeout. Didn't receive block. Appending empty block at iteration %d", timerForIteration)			
				blockChainLock.Lock()
				outLog.Printf(strconv.Itoa(client.id)+":chain lock acquired")
				blockToSend, err := client.createBlock(iterationCount)
				blockChainLock.Unlock()		
				printError("Iteration: " + strconv.Itoa(iterationCount), err)
				if(err==nil){
					sendBlock(*blockToSend)
				}

			}	
	}

}

func Intersection(a, b []int) (c []int) {
      m := make(map[int]bool)

      for _, item := range a {
              m[item] = true
      }

      for _, item := range b {
              if _, ok := m[item]; ok {
                      c = append(c, item)
              }
      }
      return
}