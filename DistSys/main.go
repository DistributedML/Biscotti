package main

import(
	"fmt"
	"bufio"
	"os"
	"strconv"
	"net"
	"net/rpc"	
	"bytes"
	"encoding/gob"
	"time"
	"log"
	// "encoding/binary"
	// "math"
	"sync"
	"errors"
	// "encoding/csv"
	// "encoding/csv"
	// "kniren/gota"
	// "io"
	// "container/list"
	"github.com/kniren/gota/dataframe"
	"github.com/sbinet/go-python"
	// "gonum.org/v1/gonum/mat"
	"github.com/DistributedClocks/GoVector/govec"

)

// RPC doesn't return. Thats the problem.

type Message struct {
	Type        string
	UpdateData  Update
	Block       Block
	// RequestData Request
	// AckData     Ack
}

type Ack struct {
	Iteration int
}

type Peer int




const (
	basePort	int 			= 8000
	myIP		string 			= "127.0.0.1:"
	verifierIP	string  		= "127.0.0.1:"
	timeoutNS 	time.Duration 	= 10000000000
)

var(
	
	datasetName 	string
	verifier 		bool
	ensureRPC  		chan error
	myPort			string
	portsToConnect  []string
	clusterPorts    []string
	client 			Honest
	iterationCount  = -1
	updateLock 		sync.Mutex
	blockLock		sync.Mutex
	boolLock		sync.Mutex
	convergedLock	sync.Mutex
	verifierLock 	sync.Mutex

	numFeatures 	int
	minClients		int
	pulledGradient 	[]float64
	deltas 			[]float64
	updates			[]Update
	updateSent		bool
	converged		bool

	numberOfNodes 	int


	errLog          *log.Logger = log.New(os.Stderr, "[err] ", log.Lshortfile|log.LUTC|log.Lmicroseconds)
	outLog          *log.Logger = log.New(os.Stderr, "[peer] ", log.Lshortfile|log.LUTC|log.Lmicroseconds)


	logger          *govec.GoLog
	didReceiveBlock  bool
	staleUpdateError error = errors.New("Stale Update")
)

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func amVerifier(nodeNum int) bool{

	//THIS WILL CHANGE AS OUR VRF APPROACH MATURES
	if((iterationCount%numberOfNodes) == client.id){
		return true
	}else{
		return false
	}

}

func main() {

	//Parsing arguments nodeIndex, numberOfNodes, datasetname
	//TODO: CLean this up a little Use flags used in the peer to peer blockchain tutorial

	//Known Issues: RPC's doesn't get called on the other side.Why? God Why?

	nodeNum, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Println("First argument should be index of node")
		return
	}
	numberOfNodes, err = strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Println("Second argument should be the total number of nodes")
		return
	}	
	datasetName := os.Args[3]


	logger = govec.InitGoVector(os.Args[1], os.Args[1])	
	gob.Register(&net.TCPAddr{})


	// getports of all other clients in the system
	myPort = strconv.Itoa(nodeNum + basePort)
	for i := 0; i < numberOfNodes; i++ {
		if strconv.Itoa(basePort+i) == myPort {
			continue
		}
		clusterPorts = append(clusterPorts, strconv.Itoa(basePort+i))
	}

	//Initialize a honest client	
	client = Honest{id: nodeNum, blockUpdates: make([]Update, 0, 5)}
	
	
	// Reading data and decalring some global locks to be used later 
	client.initializeData(datasetName, numberOfNodes)
	converged = false
	verifier = false

	updateLock = sync.Mutex{}
	blockLock = sync.Mutex{}
	boolLock = sync.Mutex{}
	convergedLock = sync.Mutex{}
	verifierLock = sync.Mutex{}
	ensureRPC = make(chan error)


	// Initializing RPC Server
	peer := new(Peer)
	peerServer := rpc.NewServer()
	peerServer.Register(peer) 


	prepareForNextIteration()		

	
	// Send Updates and Serve incoming messages at the same time
	go messageListener(peerServer, myPort)
	messageSender(clusterPorts)



}

func prepareForNextIteration() {
	
	// End if converged

	fmt.Println("RPC 11")
	convergedLock.Lock()
	if(converged){
		
		convergedLock.Unlock()		
		time.Sleep(1000 * time.Millisecond)	
		client.bc.PrintChain()
		os.Exit(1)
	}
	convergedLock.Unlock()
	fmt.Println("RPC 12")

	// If you were the verifier, empty update buffer
	boolLock.Lock()

	fmt.Println("RPC 13")

	if(verifier){
		updateLock.Lock()
		client.flushUpdates(numberOfNodes)
		updateLock.Unlock()
	}

	fmt.Println("RPC 14")
	iterationCount++
	verifier = amVerifier(client.id)
	fmt.Println("RPC 15")
	if(verifier){
		fmt.Println("I am verifier")
		updateSent = true	
	}else{		
		fmt.Println("I am not verifier")
		updateSent = false
	}	

	boolLock.Unlock()
	fmt.Println("RPC 16")





	portsToConnect = make([]string, len(clusterPorts))
	copy(portsToConnect, clusterPorts)
}

func getData(filePath string) dataframe.DataFrame{

	// Read data into the dataframe from the given filepath
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

	// Divide the dataset equally among the number of nodes

	var dividedData []dataframe.DataFrame
	indexes := make([]int, 0)

	var stepsize int
	start:= 0
	end :=  0
	
	stepsize = data.Nrow()/numberOfNodes

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

func createCSVs(nodeData dataframe.DataFrame, datasetName string, nodeID int){
	
	// create a CSV for your part of the dataset
	filename :=  datasetName + strconv.Itoa(nodeID) + ".csv"
	file, err := os.Create(datasetPath + filename)
	check(err)
	nodeData.WriteCSV(bufio.NewWriter(file))
}

func messageListener(peerServer *rpc.Server, port string) {
	
	// Listen for incoming connections and serve them

	l, e := net.Listen("tcp", myIP+port)
	handleErrorFatal("listen error", e)
	
	outLog.Printf("Peer started. Receiving on %s\n", port)

	for {
		conn, _ := l.Accept()
		outLog.Printf("Accepted new Connection")
		go peerServer.ServeConn(conn)
	}

}

func (s *Peer) VerifyUpdate(update Update, _ignored *bool) error {

	
	fmt.Printf("Got update message, iteration %d\n", update.Iteration)

	// boolLock.Lock()

	if update.Iteration < iterationCount {
		handleErrorFatal("Update of previous iteration received", staleUpdateError)
		return staleUpdateError
	}		

	for update.Iteration > iterationCount {
		// A crappy way to block until our iteration matches
		fmt.Printf("Blocking. Got update for %d, I am at %d\n", update.Iteration, iterationCount)
		time.Sleep(1000 * time.Millisecond)
	}

	go processUpdate(update)

	// fmt.Println("RPC: 3")


	// fmt.Println("RPC: 7")

	
	// fmt.Println("RPC: 8")

	return nil

}

func processUpdate(update Update){

	updateLock.Lock() 
	fmt.Println("RPC: 41")
	numberOfUpdates := client.addBlockUpdate(update) 
	updateLock.Unlock()

	fmt.Println(numberOfUpdates) 		

	// Sometimes runs until here and then stops. Its either not going inside the 
	
	fmt.Println("RPC: 41")


	if(numberOfUpdates == (numberOfNodes - 1)){

		// sometimes blocks on this
		fmt.Println("RPC: 42")

		blockToSend := client.createBlock(iterationCount)
		
		fmt.Println("RPC: 43")

		// updateLock.Unlock()		
		// If iteration ends, GoRourtine to make the relevant rpc calls for registering block	
		sendBlock(blockToSend)

		fmt.Println("RPC: 44")

	}

	fmt.Println("RPC: 45")



}

func sendBlock(block Block){

	// Send a Block to each individual client in the network

	fmt.Printf("Sending block. Iteration: %d\n", block.Data.Iteration)


	// create a thread for separate 
	for _, port := range clusterPorts {
		
		fmt.Println(port)
		go callRegisterBlockRPC(block, port)								
		fmt.Println("RPC: 4")
		
	}

	//check for convergence and move to the new iteration	

	fmt.Println("RPC: 5")
	
	convergedLock.Lock()	
	fmt.Println("RPC: 6")

	converged = client.checkConvergence()	
	convergedLock.Unlock()
	fmt.Println("RPC: 7")


	ensureRPCCallsReturn()
	prepareForNextIteration()

}

func ensureRPCCallsReturn(){
	
	for i := 0; i < (numberOfNodes-1); i++ {
		<- ensureRPC
		fmt.Println("RPC done: %i\n")
	}
}

func callRegisterBlockRPC( block Block, port string){

	// var ign bool 	
	// err := conn.Call("Peer.RegisterBlock", block ,&ign)
	// handleErrorFatal("Sending Block failed", err)
	// conn.Close()

	var ign bool	
	c := make(chan error)
	fmt.Println("RPC: 1")
	conn, er := rpc.Dial("tcp", myIP+port) // sometimes gets stuck on this.Unexpected EOF cause
	fmt.Println("RPC: 2")		
	exitOnError("rpc Dial",er)	
	defer conn.Close()
	fmt.Println("RPC: 3")		
	
	go func() { c <- conn.Call("Peer.RegisterBlock", block ,&ign) } ()
	select {
	  case err := <-c:

		fmt.Println("RPC successful")		
		handleErrorFatal("Error in sending update",err)		
		ensureRPC <- err		
		
	    // use err and result
	  case <-time.After(timeoutNS):

	  	fmt.Println("Timed out")
	    callRegisterBlockRPC(block,port)
	}

}


func (s *Peer) RegisterBlock(block Block, _ignored *bool) error {

	fmt.Printf("Got block message, iteration %d\n",  block.Data.Iteration)
	for block.Data.Iteration > iterationCount {
		// A crappy way to block until our iteration matches
		fmt.Printf("Blocking. Got block for %d, I am at %d\n", block.Data.Iteration, iterationCount)
		time.Sleep(1000 * time.Millisecond)
	}
	go addBlockToChain(block)	// can make this a goroutine
	return nil

}

func addBlockToChain(block Block){

	client.bc.AddBlockMsg(block)		
	
	//This might cause more problems than solutions
	if(block.Data.Iteration == iterationCount){
		boolLock.Lock()
		updateSent = true
		boolLock.Unlock()
	}
	

	convergedLock.Lock()
	converged = client.checkConvergence()
	convergedLock.Unlock()
	prepareForNextIteration()

}


func messageSender(ports []string) {
	
	//Continous for loop checking if update needs to be sent.
	// If update needs to be sent, compute update and make RPC call

	// replace this with channels if possible

	for{





		if(verifier){
			
			time.Sleep(100 * time.Millisecond)
			continue;
		}

		boolLock.Lock()
		
		if(!updateSent){

			
			fmt.Printf("Computing Update\n")

			client.computeUpdate(iterationCount, datasetName)

			fmt.Printf("Computed Update\n")

			portsToConnect = VRF(iterationCount)

			fmt.Printf("Computed Verifier\n")
			
			for _, port := range portsToConnect {	

				go sendUpdateToVerifier(port)
				
				fmt.Printf("RPC Called\n")
				
				if(iterationCount == client.update.Iteration){
					updateSent = true
				}					

			}

			fmt.Printf("Outside Loop\n")

			boolLock.Unlock()


			
		}else{

			boolLock.Unlock()
			fmt.Println("Yielding from sending thread")
			time.Sleep(100 * time.Millisecond)
		
		}





	}


}

func sendUpdateToVerifier( port string){

	var ign bool	
	c := make(chan error)

	conn, err := rpc.Dial("tcp", verifierIP + port)
	defer conn.Close()

	handleErrorFatal("Unable to connect to verifier",err)
	fmt.Printf("Connected to Verifier. Sending Update, Iteration:%d\n", client.update.Iteration)
	
	go func() { c <- conn.Call("Peer.VerifyUpdate", client.update, &ign) } ()
	select {
	  case err := <-c:
		fmt.Println("RPC successful")
		handleErrorFatal("Error in sending update",err)		
	    // use err and result
	  case <-time.After(timeoutNS):

	  	conn.Close()
	  	fmt.Println("Timed out")
	    sendUpdateToVerifier(port)
	}

}

func VRF(iterationCount int) []string{

	// THIS WILL CHANGE AS THE VRF IMPLEMENTATION CHANGES
	verifiers := make([]string, 1)
	verifiers[0] = strconv.Itoa(basePort + (iterationCount % numberOfNodes))
	return verifiers	
}

func handleErrorFatal(msg string, e error) {
	if e != nil {
		errLog.Fatalf("%s, err = %s\n", msg, e.Error())
	}
}

func exitOnError(prefix string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s, err = %s\n", prefix, err.Error())
		os.Exit(1)
	}
}


// Some redundant code. This doesn't get used

func (msg Message) ToByte() []byte {

	var msgBytes bytes.Buffer 
	enc:= gob.NewEncoder(&msgBytes)       
    err := enc.Encode(msg)
    if err != nil {
        fmt.Println("encode error:", err)
    }

    return msgBytes.Bytes()

    // How to decode this thing. I will leave it here for future ref.

    // var q Q
    // err = dec.Decode(&q)
    // if err != nil {
    // Decode (receive) the value.
    //     log.Fatal("decode error:", err)
    // }
    // fmt.Printf("%q: {%d,%d}\n", q.Name, *q.X, *q.Y)

} 

func byteToMessage(buf []byte) *Message {

	// fmt.Println(buf)
	msg := Message{}
	msgBytes := bytes.NewBuffer(buf)
	dec := gob.NewDecoder(msgBytes)    
    err := dec.Decode(&msg)
    
    if err != nil {
    	fmt.Println("Decode failed")
    }
    
    fmt.Println(msg)

    // fmt.Printf(msg.UpdateData.String())

    return &msg
}




// func packetListener(conn net.Conn) {

// 	// handle messages based on whether they are updates or blocks


// 	inBuf := make([]byte, 2048)

// 	outBuf := make([]byte, 2048)


// 	n, err := conn.Read(inBuf)
// 	if err != nil {
// 		fmt.Printf("Got a reply read failure reading %d bytes.\n", n)
// 		conn.Close()
// 		os.Exit(1)
// 	}

// 	var message Message;
// 	logger.UnpackReceive("received message", inBuf[0:n], &message)

// 	switch message.Type {

// 		case "update":

// 			fmt.Printf("Got update message %d bytes, iteration %d\n", n, message.UpdateData.Iteration)

			

// 			for message.UpdateData.Iteration > iterationCount {
// 				// A crappy way to block until our iteration matches
// 				fmt.Printf("Blocking. Got update for %d, I am at %d\n", message.UpdateData.Iteration, iterationCount)
// 				time.Sleep(1000 * time.Millisecond)
// 			}

// 			updateLock.Lock() 
// 			numberOfUpdates := client.addBlockUpdate(message.UpdateData) 


// 			fmt.Println(numberOfUpdates) 
			

// 			if(numberOfUpdates == (numberOfNodes - 1)){
// 				blockToSend := client.createBlock(iterationCount)
// 				blockMsg := Message{Type: "block", Block: blockToSend}
// 				outBuf = logger.PrepareSend("Sending block to all other nodes", blockMsg)
				
				
// 				fmt.Printf("Sending block. Iteration: %d\n", blockMsg.Block.Data.Iteration)
// 				for _, port := range clusterPorts {

// 					fmt.Println(port)
// 					conn, err := net.Dial("tcp", verifierIP+port)
// 					if err != nil {
// 						fmt.Printf("Could not connect to %s\n", port)
// 						continue
// 					}
// 					n, err := conn.Write(outBuf)
// 					if err != nil {
// 						fmt.Println("Got a conn write failure writing %d bytes.", n)
// 						conn.Close()
// 						os.Exit(1)
// 					}
// 					conn.Close()						
// 				}

// 				converged = client.checkConvergence()
				
// 				prepareForNextIteration()
// 			}
// 			updateLock.Unlock()


// 		case "block":
// 			fmt.Printf("Got block message %d bytes, iteration %d\n", n, message.Block.Data.Iteration)


// 			for message.Block.Data.Iteration > iterationCount {
// 				// A crappy way to block until our iteration matches
// 				fmt.Printf("Blocking. Got block for %d, I am at %d\n", message.Block.Data.Iteration, iterationCount)
// 				time.Sleep(1000 * time.Millisecond)
// 			}

// 			client.bc.AddBlockMsg(message.Block)			


// 			converged = client.checkConvergence()

// 			prepareForNextIteration()

// 	}

// 	conn.Close()
// }


	// outBuf := make([]byte, 2048)

	// for {


	// 	if(verifier){

	// 		continue;
	// 	}

	// 	if(!updateSent){
			
	// 		boolLock.Lock()
	// 		fmt.Printf("Computing Update\n")

	// 		client.computeUpdate(iterationCount, datasetName)

	// 		portsToConnect = VRF(iterationCount)

	// 		portsToRetry := []string{}
	// 		for _, port := range portsToConnect {

	// 			conn, err := net.Dial("tcp", verifierIP+port)

	// 			if err != nil {
	// 				fmt.Printf("Could not connect to %s\n", port)
	// 				portsToRetry = append(portsToRetry, port)
	// 				continue
	// 			}

	// 			var msg Message

	// 			msg.Type = "update"
	// 			msg.UpdateData = client.update

	// 			outBuf = logger.PrepareSend("Sending update to verifier", msg)

	// 			n, err := conn.Write(outBuf)

	// 			fmt.Printf("Update sent %d bytes, Iteration: %d\n " , n, msg.UpdateData.Iteration)
	// 			conn.Close()

	// 			if err != nil {
	// 				fmt.Println("Got a conn write failure writing %d bytes.", n)
	// 				conn.Close()
	// 				os.Exit(1)
	// 			}
				
	// 		}
	// 		updateSent = true
	// 		boolLock.Unlock()		
			
	// 	}

	// }

	//Deprecated Listening code
	// myaddr, err := net.ResolveTCPAddr("tcp", myIP+port)
	// if err != nil {
	// 	fmt.Println("LISTEN ERROR")
	// }

	// ln, err := net.ListenTCP("tcp", myaddr)
	// if err != nil {
	// 	fmt.Println("Could not listen for messages")
	// }

	// // Reponse to messages. Is this the right way t
	// for {
	// 	fmt.Println("Waiting for a message...")

	// 	conn, err := ln.Accept()
	// 	if err != nil {
	// 		fmt.Println("Could not accept connection on port")
	// 	}

	// 	fmt.Println("Got a ping.")
	// 	go packetListener(conn)
		
	// }


