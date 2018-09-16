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
	"sort"
	"math"
)

// Timeout for block should be more than timeout for update because nodes should be more patients for the block to come through

//Assumption: Requires node 0 to be online first. 

const (
	basePort        int           = 8000
	verifierIP   	string        = "127.0.0.1:"
	timeoutRPC    	time.Duration = 10000000000
	timeoutUpdate 	time.Duration = 10000000000  
	timeoutBlock 	time.Duration = 15000000000  
	timeoutPeer 	time.Duration = 5000000000
	
	NUM_VERIFIERS 	int           = 3
	NUM_MINERS 		int           = 3
	NUM_NOISERS     int 		  = 2
	DEFAULT_STAKE   int 		  = 10

	VERIFIER_PRIME 	int 		  = 2
	MINER_PRIME 	int 		  = 3
	NOISER_PRIME 	int 		  = 5

	PRECISION       int 		  = 4
	POLY_SIZE 		int 		  = 10

	MAX_ITERATIONS  int 		  = 15
	EPSILON 		float64 	  = 1
	SECURE_AGG  	bool 		  = true
	NOISY_VERIF		bool 		  = true

	PRIV_PROB 		float64 	  = 0.4

)

type Peer int

type NoiseVector struct {
	noise []float64
}

var (

	//Input arguments
	datasetName   		string
	numberOfNodes 		int
	TOTAL_SHARES 		int 		  

	numberOfNodeUpdates int
	myIP                string
	myPrivateIP         string
    myPort				string
    peersFileName       string

	client 				Honest
	myVRF				VRF
	collusionThresh 	int
	unmaskedUpdates 	int
	totalUpdates	 	int
	
	allSharesReceived		chan bool
	allUpdatesReceived		chan bool
	networkBootstrapped		chan bool
	blockReceived 			chan bool
	quitRoutine 			chan bool
	portsToConnect 			[]string
	verifierPortsToConnect 	[]string
	minerPortsToConnect 	[]string
	noiserPortsToConnect 	[]string
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
	signatureError  error = errors.New("Insufficient correct signatures collected")

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

func (s *Peer) VerifyUpdate(update Update, signature *[]byte) error {

	outLog.Printf(strconv.Itoa(client.id)+":Got RONI message, iteration %d\n", update.Iteration)

	/*	// we can return the chain to the guy here instead of just leaving that guy with an error
	if update.Iteration < iterationCount {
		printError("Update of previous iteration received", staleError)
		// sender is stale, return true here and let them catch up
		return true
	}*/

	roniScore := client.verifyUpdate(update)
	outLog.Printf("RONI for update at iteration %d is %f.\n", update.Iteration, roniScore)

	if (PRIV_PROB > 0) {
		
		outLog.Printf("Accepting update!")
		updateCommitment := update.Commitment
		(*signature) = SchnorrSign(updateCommitment, client.Keys.Skey)		
		return nil	
	}

	// Roni score measures change in local training error
	if roniScore > 0.02 {
	
		outLog.Printf("Rejecting update!")		
		return roniError
	
	}else{

		outLog.Printf("Accepting update!")
		updateCommitment := update.Commitment
		(*signature) = SchnorrSign(updateCommitment, client.Keys.Skey)
		/*mySkey, _ := client.Keys.Skey.MarshalBinary()
		fmt.Println("My Skey:%s", mySkey)
		fmt.Println("My Signature:%s", (*signature))*/
		return nil	
	}	

}

// The peer receives an update from another peer if its a noiser in that round.
// The noiser peer returns their noise at the given iteration value
// Returns:
// - roniError if it fails
func (s *Peer) RequestNoise(iterationCount int, returnNoise *[]float64) error {

	outLog.Printf(strconv.Itoa(client.id)+":Got noise message, iteration %d\n", iterationCount)

	noise, err := client.requestNoise(iterationCount)
	*returnNoise = noise

	return err

}

// The peer receives an update from another peer if its a verifier in that round.
// The verifier peer takes in the update and returns immediately.
// It calls a separate go-routine for collecting updates and sending updates when all updates have been collected
// Returns:
// - StaleError if its an update for a preceding round.

func (s *Peer) RegisterSecret(share MinerPartRPC, _ignored *bool) error {

	outLog.Printf(strconv.Itoa(client.id)+":Got secret share, iteration %d\n", share.Iteration)

	// we can return the chain to the guy here instead of just leaving that guy with an error
	if share.Iteration < iterationCount {
		printError("Share of previous iteration received", staleError)
		return staleError
	}

	outLog.Printf("Length of signature:%d", len(share.SignatureList))
	outLog.Printf("Length of verifiers:%d", len(verifierPortsToConnect))

	if ((len(share.SignatureList) < len(verifierPortsToConnect)/2) || !verifySignatures(share.SignatureList, share.CommitmentUpdate)){
		printError("Share has insufficient or bogus signatures", signatureError)
		return signatureError
	}

	// Process update only called by the miner nodes
	realShare := converttoMinerPart(share) 

	go processShare(realShare)

	return nil

}

func verifySignatures(signatureList [][]byte, commitment []byte) bool{

	
	signatureVerified := false
	for i := 0; i < len(signatureList); i++ {
	
		thisSignature := signatureList[i]
		
		signatureVerified = false
		
		for i := 0; i < len(verifierPortsToConnect); i++ {
		
			verifierID := peerLookup[verifierPortsToConnect[i]]
			verifierPubKey := client.Keys.PubKeyMap[verifierID].PKG1[0]
			err := SchnorrVerify(commitment, verifierPubKey ,thisSignature)
		
			if(err == nil){

				signatureVerified = true
				break
			}
			
			// }else{
			// 	// fmt.Println("Verifier Public Key:%s", verifierPubKey)
			// 	// fmt.Println("This Signature: %s", thisSignature)
			// 	// fmt.Println("This Error: %s",err)
			// 	// fmt.Println("This commitment: %s", commitment)
			// }
			
		}
		
		if !signatureVerified{
			break
		}
	
	}

	return signatureVerified



}

// go routine to process the update received by miner nodes
func processShare(share MinerPart) {

	outLog.Printf(strconv.Itoa(client.id)+":Got share for %d, I am at %d\n", share.Iteration, iterationCount)

	for share.Iteration > iterationCount {
		outLog.Printf(strconv.Itoa(client.id)+":Blocking for stale update. Update for %d, I am at %d\n", share.Iteration, iterationCount)
		time.Sleep(2000 * time.Millisecond)
	}

	// Might get an update while I am in the announceToNetwork phase and when I come out of it the update becomes redundant
	if ((iterationCount == share.Iteration)) {

		outLog.Printf(strconv.Itoa(client.id)+":Appending secret share, iteration %d\n", share.Iteration)	

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

func (s *Peer) GetMinerPart(nodeList []int, myMinerPartRPC *MinerPartRPC) error {

	outLog.Printf(strconv.Itoa(client.id) + ":Giving my aggregated share")
	
	myMinerPart := MinerPart{}
	// Might need this condition
	if(miner && (client.secretList[nodeList[0]].Iteration == iterationCount)){

		myMinerPart = client.secretList[nodeList[0]]

		for i := 1; i < len(nodeList); i++ {
			
			myMinerPart = aggregateSecret(myMinerPart, client.secretList[nodeList[i]])
		}

		*(myMinerPartRPC) = converttoRPC(myMinerPart)

	
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

	vIDs, mIDs, _, _ := myVRF.getVRFRoles(stakeMap, client.bc.getLatestBlockHash(), 
		NUM_VERIFIERS, NUM_MINERS, numberOfNodes)

	nIDs, _, _ := myVRF.getVRFNoisers(stakeMap, client.bc.getLatestBlockHash(), 
		client.id, NUM_NOISERS, numberOfNodes)

	outLog.Printf("Verifiers are %v", vIDs)
	outLog.Printf("Miners are %v", mIDs)
	outLog.Printf("Noisers are %v", nIDs)

	for _, id := range vIDs {
		roleMap[id] *= VERIFIER_PRIME
	}

	for _, id := range mIDs {
		roleMap[id] *= MINER_PRIME
	}

	for _, id := range nIDs {
		roleMap[id] *= NOISER_PRIME
	}

	return roleMap
}

// Convert the roleIDs to verifier/miner strings 
func getRoleNames(iterationCount int) ([]string, []string, []string, int) {

	verifiers := make([]string, 0)
	miners := make([]string, 0)
	noisers := make([]string, 0)
	numVanilla := 0

    // Find the address corresponding to the ID.
    // TODO: Make fault tolerant
    // TODO: Maybe implement inverted index
    for address, ID := range peerLookup {

    	// A bit messy. The contributors are all non verifier, non miner.
    	if (roleIDs[ID] == 1 || roleIDs[ID] == NOISER_PRIME) {
    		numVanilla++
    	}
        
    	if (roleIDs[ID] % VERIFIER_PRIME) == 0 {
    		verifiers = append(verifiers, address)
    	}

    	if (roleIDs[ID] % MINER_PRIME) == 0 {
    		miners = append(miners, address)
    	}

    	if (roleIDs[ID] % NOISER_PRIME) == 0 {
    		noisers = append(noisers, address)
    	}

    }

	return verifiers, miners, noisers, numVanilla

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
	TOTAL_SHARES = int(math.Ceil(float64(POLY_SIZE)/float64(NUM_MINERS)))*NUM_MINERS

	// Reading data and declaring some global locks to be used later
	collusionThresh = int(math.Ceil(float64(numberOfNodes) * (1.0 - PRIV_PROB)))
	
	if NOISY_VERIF && (nodeNum < collusionThresh) {

		client.initializeData(datasetName, numberOfNodes, EPSILON)	
	
	} else {
		
		client.initializeData(datasetName, numberOfNodes, 0)	
	
	}
	
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

func isCollusionAttack(verifiers []string, noisers []string) bool{

	verifierColludes := false

	for i := 0; i < len(verifiers); i++ {
		if(peerLookup[verifiers[i]] >= collusionThresh){

			verifierColludes = true
			break

		}
	}

	if(!verifierColludes){
		return false
	}

	noiserCollusion := true

	for i := 0; i < len(noisers); i++ {
	
		if(peerLookup[noisers[i]] < collusionThresh){

			noiserCollusion = false	
			break
		}	
	
	}

	return noiserCollusion

}

// At the start of each iteration, this function is called to reset shared global variables
// based on whether you are a verifier or not.

func prepareForNextIteration() {

	convergedLock.Lock()

	if converged {

		if (PRIV_PROB > 0){
			fmt.Println(unmaskedUpdates, totalUpdates, PRIV_PROB)
		}
		
		convergedLock.Unlock()
		time.Sleep(1000 * time.Millisecond)
		client.bc.PrintChain()
		os.Exit(1)
	
	}else{

		if iterationCount > MAX_ITERATIONS {
			
			if (PRIV_PROB > 0){
				fmt.Println(unmaskedUpdates, totalUpdates, PRIV_PROB)
			}
			client.bc.PrintChain()
			os.Exit(1)	
		
		}
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

	verifierPortsToConnect, minerPortsToConnect, 
		noiserPortsToConnect, numberOfNodeUpdates = getRoleNames(iterationCount)

	if (PRIV_PROB > 0) {
		
		totalUpdates = totalUpdates + len(client.bc.Blocks[len(client.bc.Blocks) - 1].Data.Deltas)

		if (isCollusionAttack(verifierPortsToConnect, noiserPortsToConnect)) {

				unmaskedUpdates = unmaskedUpdates + 1
		}
	}



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
		// startBlockDeadlineTimer(iterationCount)
	
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

		outLog.Printf("As miner, I expect %d updates, I have gotten %d", numberOfNodeUpdates, numberOfUpdates)

		//send signal to start sending Block if all updates Received. Changed this from numVanilla stuff
		if numberOfUpdates == (numberOfNodeUpdates) {			
			outLog.Printf(strconv.Itoa(client.id)+":All updates for iteration %d received. Notifying channel.", iterationCount)	
			allUpdatesReceived <- true 		 
		}	
	
	}

}

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

			addBlockToChain(block)

		}

		return
	}
			
	// // if not empty and not verifier send signal to channel. Not verifier required because you are not waiting for a block if you are the verifier and if you receive an empty block and if you are currently busy bootstrapping yourself. 
	// if(len(block.Data.Deltas) != 0 && !verifier && iterationCount >= 0) {
	// 	blockReceived <- true
	// }

	
	addBlockToChain(block)

}

// go-routine to process a block received and add to chain. 
// Move to next iteration when done

func addBlockToChain(block Block) {

	// Add the block to chain
	blockChainLock.Lock()
	
	outLog.Printf(strconv.Itoa(client.id)+":Adding block for %d, I am at %d\n", 
		block.Data.Iteration, iterationCount)

	err := client.addBlock(block)
	
	// Update the stake in the system
	if len(block.StakeMap) > 0 {
		stakeMap = block.StakeMap
		outLog.Printf("New stake map: %v", stakeMap)
	}

	blockChainLock.Unlock()

	if ((block.Data.Iteration == iterationCount) && (err == nil)){
	
		// If block is current, notify channel waiting for it
		if(len(block.Data.Deltas) != 0 && updateSent && !verifier  && iterationCount >= 0) {
			
			if SECURE_AGG {

				outLog.Printf(strconv.Itoa(client.id)+":Sending block to channel")				
				blockReceived <- true
				
			} else {

				if(!miner) {

					outLog.Printf(strconv.Itoa(client.id)+":Sending block to channel")
					blockReceived <- true
				}

				if(miner && getLeaderAddress() == (myIP+myPort)) {
					outLog.Printf("Blocking here")
					quitRoutine <- true
				}
			}		
		}

		if SECURE_AGG {
			
			if(miner && getLeaderAddress() == (myIP+myPort) && len(block.Data.Deltas) == 0){
				outLog.Printf("Blocking here")
				quitRoutine <- true
			}
		
		}
		

		boolLock.Unlock()

		outLog.Printf("Bool lock released")
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

			client.computeUpdate(iterationCount)

			outLog.Printf(strconv.Itoa(client.id)+":Getting noise from %s\n", noiserPortsToConnect)

			noise := requestNoiseFromNoisers(noiserPortsToConnect)

			outLog.Printf("Noise:%s", noise)

			if (len(noise) > 0) {
				
				noiseDelta := make([]float64, len(noise))
				
				outLog.Printf("Update Delta:%s",len(client.update.Delta))
				outLog.Printf("Noise:%s",len(noise))
				outLog.Printf("noiseDelta:%s",len(noiseDelta))

				for i := 0; i < len(noise); i++ {

					noiseDelta[i] = client.update.Delta[i] + noise[i]
				}

				// The default is if no noise being used, NoisedDelta will just be Delta.
				client.update.Noise = noise
				client.update.NoisedDelta = noiseDelta

			}

			outLog.Printf("Sending update to verifiers")
			signatureList, approved := sendUpdateToVerifiers(verifierPortsToConnect)			 

			// outLog.Printf("Signature List: %s", signatureList)

			if approved {
				
				// break update down into secrets
				// send secrets to miners

				outLog.Printf("Sending update to miners")
				client.update.SignatureList = signatureList

				if SECURE_AGG {
					sendUpdateSecretsToMiners(minerPortsToConnect)									
				}else{
					sendUpdateToMiners(minerPortsToConnect)
				}			
			

			}else{

				go startBlockDeadlineTimer(iterationCount)
			}

			if iterationCount == client.update.Iteration {
					updateSent = true
			}


			boolLock.Unlock()

		} else {

			boolLock.Unlock()
			time.Sleep(100 * time.Millisecond)

		}
	}
}

// RPC call to send block to one peer
func requestNoiseFromNoisers(addresses []string) []float64 {

	var noiseVec []float64

	// Just return an empty noise vector, case is handled downstream.
	if !NOISY_VERIF {
		return noiseVec
	}

	noisesReceived := 0.0
	noiseVec = make([]float64, client.ncol)
	c := make(chan error)

	for _, address := range addresses {

		var thisVec []float64
		conn, err := rpc.Dial("tcp", address)
		printError("Unable to connect to noiser", err)
	
		if(err == nil){
			
			defer conn.Close()
			outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Noiser. Iteration:%d\n", client.update.Iteration)
			go func() { c <- conn.Call("Peer.RequestNoise", client.update.Iteration, &thisVec) } ()
			
			select {
			case noiseError := <-c:
				
				printError("Error in sending update", err)
				if (noiseError == nil) {
		
					outLog.Printf(strconv.Itoa(client.id)+":Got noise. Iteration:%d\n", client.update.Iteration)
					noisesReceived++

					for i := 0; i < len(thisVec); i++ {
						noiseVec[i] += thisVec[i]
					}

				}

			// use err and result
			case <-time.After(timeoutRPC):
				outLog.Printf(strconv.Itoa(client.id)+":RPC Call timed out.")
				continue
			}
		
		} else {

			outLog.Printf("GOT NOISER ERROR")
			time.Sleep(1000 * time.Millisecond)

			continue
		}

	}

	// Take the average of the noise to preserve privacy budget
	for i := 0; i < len(noiseVec); i++ {
		noiseVec[i] /= noisesReceived
	}

	return noiseVec

}

// Make RPC call to send update to verifier
// If you cant connect to verifier or verifier fails midway RPC, then append an empty block and move on
// Start timer for receiving registering block

func sendUpdateToVerifiers(addresses []string) ([][]byte ,bool) {

	signatureList := make([][]byte, 0)
	c := make(chan error)
	verified := false
	verifiersOnline := false

	for _, address := range addresses {

		conn, err := rpc.Dial("tcp", address)
		printError("Unable to connect to verifier", err)
		
		if(err == nil){
			
			defer conn.Close()
			signature := []byte{}
			outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Verifier. Sending Update, Iteration:%d\n", client.update.Iteration)
			go func() { c <- conn.Call("Peer.VerifyUpdate", client.update, &signature) }()
			select {
			case verifierError := <-c:
				
				printError("Error in sending update", err)
				verifiersOnline  = true
				if (verifierError == nil) {
					outLog.Printf(strconv.Itoa(client.id)+":Update verified. Iteration:%d\n", client.update.Iteration)
					signatureList = append(signatureList, signature)
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

	if (len(signatureList) >= (len(verifierPortsToConnect)/2)) {

		verified = true

	}

	// Verification totally failed. Create empty block and send
	if !verifiersOnline {
		outLog.Printf(strconv.Itoa(client.id)+":Will try and create an empty block")
		blockChainLock.Lock()
		blockToSend, err := client.createBlock(iterationCount, stakeMap)
		blockChainLock.Unlock()		
		printError("Iteration: " + strconv.Itoa(iterationCount), err)
		if(err==nil){
			outLog.Printf(strconv.Itoa(client.id)+":Sending an empty block")
			go sendBlock(*blockToSend)
		}
	} 

	return signatureList, verified

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
		blockToSend, err := client.createBlock(iterationCount, stakeMap)
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

	mined := true

	// generate secrets here
	minerSecrets := generateMinerSecretShares(client.update.Delta, PRECISION, client.Keys.CommitmentKey, NUM_MINERS, POLY_SIZE, TOTAL_SHARES)

	// outLog.Printf("My secret share:%s", minerSecrets)

	// for i := 0; i < len(minerSecrets); i++ {
		// outLog.Printf("My secret share 10:%s", minerSecrets[i].PolyMap[10].Secrets)		
		// outLog.Printf("My secret share 20:%s", minerSecrets[i].PolyMap[20].Secrets)
		// outLog.Printf("My secret share 20:%s", minerSecrets[i].PolyMap[25].Secrets)		
	// }

	// fmt.Println(minerSecrets)
	// fmt.Println(minerSecrets[0].PolyMap[10].Secrets)
	// fmt.Println(minerSecrets[1].PolyMap[10].Secrets)
	// os.Exit(1)


	// // TODO: For now, the first miner that gets the block done is good enough.
	// // We will need to use shamir secrets here later

	sort.Strings(addresses)

	minerIndex := 0

	for _, address := range addresses {

		// if !mined {

			conn, err := rpc.Dial("tcp", address)
			printError("Unable to connect to miner", err)
			
			if (err == nil) {
				
				minerSecrets[minerIndex].Iteration = client.update.Iteration
				minerSecrets[minerIndex].NodeID = client.id
				minerSecrets[minerIndex].SignatureList = client.update.SignatureList
				
				defer conn.Close()

				minerSecretRPC := converttoRPC(minerSecrets[minerIndex])				

				outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Miner. Sending Update Share, Iteration:%d\n", client.update.Iteration)

				go func() { c <- conn.Call("Peer.RegisterSecret", minerSecretRPC, &ign) }()
				
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
				mined = false
				continue
			}

		// }

		minerIndex++

	}

	// Couldn't mine the block. Send empty block. // Why do we need this?
	if !mined {
		outLog.Printf(strconv.Itoa(client.id)+":Will try and create an empty block")
		blockChainLock.Lock()
		blockToSend, err := client.createBlock(iterationCount, stakeMap)
		blockChainLock.Unlock()		
		printError("Iteration: " + strconv.Itoa(iterationCount), err)
		if(err==nil){
			outLog.Printf(strconv.Itoa(client.id)+":Sending an empty block")
			go sendBlock(*blockToSend)
		}
	}

}

// Timer started by the miner to set a deadline for receiving updates
func startUpdateDeadlineTimer(timerForIteration int){
	
	select {
		
		case <- allUpdatesReceived:
			outLog.Printf(strconv.Itoa(client.id)+":All Updates Received for timer on %d. I am at %d. Preparing to send block..", 
				timerForIteration, iterationCount)

		case <- time.After(timeoutUpdate):
			outLog.Printf(strconv.Itoa(client.id)+":Timeout. Didn't receive expected number of updates. Preparing to send block. Iteration: %d..", iterationCount)
		
		case <- quitRoutine:
			outLog.Printf(strconv.Itoa(client.id)+"Already appended block. Quitting routine. Iteration: %d..", iterationCount)			
			return 
	}
	
	if (timerForIteration == iterationCount) {
		
		leaderAddress := getLeaderAddress()	

		if (myIP+myPort) == leaderAddress {
			
			if (len(client.blockUpdates) > 0 && (myIP+myPort) == leaderAddress) {
		
				outLog.Printf(strconv.Itoa(client.id)+":Acquiring chain lock")
				blockChainLock.Lock()
				
				outLog.Printf(strconv.Itoa(client.id)+":chain lock acquired")
				blockToSend, err := client.createBlock(iterationCount, stakeMap)
				
				blockChainLock.Unlock()		
				printError("Iteration: " + strconv.Itoa(iterationCount), err)
				
				if (err == nil) {
					sendBlock(*blockToSend)
				}

			} else {

				outLog.Printf("Timer is for %d", timerForIteration)
				outLog.Printf("I am on %d", iterationCount)

				outLog.Printf(strconv.Itoa(client.id)+":Received no updates from peers. I WILL DIE")
				os.Exit(1)
			}
				
		} else {

			go startBlockDeadlineTimer(iterationCount)
		
		}

	}
	// An old timer was triggered, try to catch up
	// } else {
	// 	time.Sleep(1000 * time.Millisecond)
	// 	outLog.Printf(strconv.Itoa(client.id)+":Forwarding timer ahead.")
	// 	allUpdatesReceived <- true
	// }

}

// Timer started by the verifier to set a deadline until which he will receive updates
func getLeaderAddress() string{

	maxId := -1
	maxAddress := "Nil"

	for _, address := range minerPortsToConnect{

		thisID := peerLookup[address]
		
		if (thisID > maxId) {
			maxAddress = address
			maxId = thisID
		}

	}

	return maxAddress

}
func startShareDeadlineTimer(timerForIteration int){
	
	select {
		
		case <- allSharesReceived:
			outLog.Printf(strconv.Itoa(client.id)+":All Shares Received for timer on %d. I am at %d. Preparing to send block..", 
				timerForIteration, iterationCount)

		case <- time.After(timeoutUpdate):
			outLog.Printf(strconv.Itoa(client.id)+":Timeout. Didn't receive expected number of shares. Preparing to send block. Iteration: %d..", iterationCount)

		case <- quitRoutine:
			outLog.Printf(strconv.Itoa(client.id)+"Already appended block. Quitting routine. Iteration: %d..", iterationCount)			
			return	
	
	}
	
	// If I am on the current iteration and am the CHOSEN ONE among the miners
	if (timerForIteration == iterationCount) {

		leaderAddress := getLeaderAddress()

		outLog.Printf("MinerAddress:%s", leaderAddress)
		outLog.Printf("My Address:%s", myIP+myPort)


		if ((myIP+myPort) == leaderAddress && len(client.secretList) > 0) {

			minerMap, finalNodeList := getNodesList(minerPortsToConnect)

			sharesPerMiner := TOTAL_SHARES / NUM_MINERS

			// collected sufficient shares and there are more than one 
			if ((sharesPerMiner * len(minerMap) >= POLY_SIZE) && len(finalNodeList) > 1) {

				client.aggregatedSecrets = getSecretShares(minerMap, finalNodeList)

				outLog.Printf(strconv.Itoa(client.id)+":Acquiring chain lock")
				blockChainLock.Lock()
			
				outLog.Printf(strconv.Itoa(client.id)+":chain lock acquired")
				
				// //TODO:
				blockToSend, err := client.createBlockSecAgg(iterationCount, finalNodeList, stakeMap)
			
				blockChainLock.Unlock()		

				printError("Iteration: " + strconv.Itoa(iterationCount), err)
			
				if (err == nil) {
					sendBlock(*blockToSend)
				}			
				
			}else{

				outLog.Printf(strconv.Itoa(client.id)+":Creating empty block2")
				// create empty block
				outLog.Printf(strconv.Itoa(client.id)+":Creating empty block")
				dummyNodeList := make([]int,0)
				//create empty block
				blockChainLock.Lock()				
				outLog.Printf(strconv.Itoa(client.id)+":chain lock acquired")					
				// //TODO:
				blockToSend, err := client.createBlockSecAgg(iterationCount, dummyNodeList, stakeMap)				
				blockChainLock.Unlock()	
				outLog.Printf(strconv.Itoa(client.id)+":chain lock released")

				if (err == nil) {
					sendBlock(*blockToSend)
				}
			
			}
		

		} else {

			if ((myIP+myPort) != leaderAddress) {

				go startBlockDeadlineTimer(iterationCount)							
			
			}else{
				
				outLog.Printf(strconv.Itoa(client.id)+":Creating empty block3")					
				// create empty block
				outLog.Printf(strconv.Itoa(client.id)+":Creating empty block")
				dummyNodeList := make([]int,0)
				//create empty block
				blockChainLock.Lock()				
				outLog.Printf(strconv.Itoa(client.id)+":chain lock acquired")					
				// //TODO:
				blockToSend, err := client.createBlockSecAgg(iterationCount, dummyNodeList, stakeMap)				
				blockChainLock.Unlock()	
				if (err == nil) {
					sendBlock(*blockToSend)
				}

			}

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

	// aggregating and appending my own list
	myMinerPart := client.secretList[nodeList[0]]

	for i := 1; i < len(nodeList); i++ {
		myMinerPart = aggregateSecret(myMinerPart, client.secretList[nodeList[i]])
	}

	outLog.Printf("My miner share:%s", myMinerPart)

	minerShares = append(minerShares, myMinerPart)	

	for address, _ := range minerList {
		
		if address == myIP+myPort {
			continue
		}

		outLog.Printf(strconv.Itoa(client.id)+":Calling %s", address)
		thisMinerSecret, err := callGetMinerShareRPC(address, nodeList)		    
		
		if err == nil{
			minerShares = append(minerShares, thisMinerSecret)
		}

	}

	return minerShares
	
}


func callGetMinerShareRPC(address string, nodeList []int) (MinerPart, error){

	thisMinerPartRPC := MinerPartRPC{}
	thisMinerPart := MinerPart{}
	
	c := make(chan error)

	// TODO: For now, the first miner that gets the block done is good enough.
	// We will need to use shamir secrets here later

	conn, err := rpc.Dial("tcp", address)
	printError("Unable to connect to fellow miner", err)
	
	if (err == nil) {
		
		defer conn.Close()
		outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Fellow Miner. Getting MinerShare, Iteration:%d\n", client.update.Iteration)
		go func() { c <- conn.Call("Peer.GetMinerPart", nodeList, &thisMinerPartRPC) }()
		select {
		case err := <-c:
			
			printError("Error in getting miner part", err)
			thisMinerPart = converttoMinerPart(thisMinerPartRPC)
			if(err==nil){
				outLog.Printf(strconv.Itoa(client.id)+":Got secret share. Iteration:%d\n", iterationCount)
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
	
	listOfUpdates := make(map[string][]int)

	listOfUpdates[myIP+myPort] = make([]int, 0, len(client.secretList))

	for nodeID, _ := range client.secretList {
		listOfUpdates[myIP+myPort] = append(listOfUpdates[myIP+myPort], nodeID)
	}


	// popuate list of updates with node list of each online and synchronous miner

	for _, address := range minerList {
        
        if (address == myIP+myPort){
        	continue
        }

        outLog.Printf(strconv.Itoa(client.id)+":Calling %s", address)
		thisNodesList, err := callGetUpdateListRPC(address)
		if ((err == nil) && (len(thisNodesList) > 0)) {
			listOfUpdates[address] = thisNodesList	
			
		}

	}

	// find node ids whose updates are available with all online miners.

	intersectionList := listOfUpdates[myIP+myPort]

	for _, nodeList := range listOfUpdates {

		intersectionList = Intersection(intersectionList, nodeList)

	}

	outLog.Printf("Intersection List: %s", intersectionList)
	outLog.Printf("Node List: %s",listOfUpdates)

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
		outLog.Printf(strconv.Itoa(client.id)+":Making RPC Call to Fellow Miner. Getting NodeList, Iteration:%d\n", iterationCount)
		go func() { c <- conn.Call("Peer.GetUpdateList", iterationCount, &nodeList) }()
		select {
		case err := <-c:
			
			printError("Error in sending update", err)
			
			if(err==nil){
				outLog.Printf(strconv.Itoa(client.id)+":List received. List:%s. Iteration:%d\n", nodeList, iterationCount)
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
				blockToSend, err := client.createBlock(iterationCount, stakeMap)
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