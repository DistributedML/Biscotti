package main

import(
	"fmt"
	"bufio"
	"os"
	"strconv"
	"net"	
	"bytes"
	"encoding/gob"
	"time"
	// "encoding/binary"
	// "math"
	"sync"
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

type Request struct {
	Iteration int
}


const (
	basePort	int 	= 9000
	myIP		string 	= "127.0.0.1:"
	verifierIP	string  = "127.0.0.1:"
)

var(
	
	datasetName 	string
	verifier 		bool
	// batch_size		int
	

	myPort			string
	portsToConnect  []string
	clusterPorts    []string
	client 			Honest
	iterationCount  = -1
	updateLock 		sync.Mutex
	blockLock		sync.Mutex
	boolLock		sync.Mutex

	numFeatures 	int
	minClients		int
	pulledGradient 	[]float64
	deltas 			[]float64
	updates			[]Update
	updateSent		bool
	// DataFrame dataframe.DataFrame
	// thisRecord []string
	// pyLogModule       *python.PyObject
	// pyLogInitFunc     *python.PyObject
	// pyLogPrivFunc     *python.PyObject
	// pyNumFeatures 	  *python.PyObject

	numberOfNodes 	int

	logger          *govec.GoLog
	didReceiveBlock  bool
)

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func amVerifier(nodeNum int) bool{

	//some trash condition. TODO: get rid of it
	if((iterationCount%numberOfNodes) == client.id){
		return true
	}else{
		return false
	}

}

func main() {

	// Goal:

	// if am verifier node, don't generate update
	// just listen for incoming connections in this iteration
	// when I get a connection put them together in a block and send to everybody (if you have atleast one update you do that)
	// else
	// generateUpdate using my dataset and send to verifier node

	// What if a client joins in a later iteration? How does he know which iteration it is and
	// which// just wait for the next block
	
	//send to verifier node
	//verifier code 


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

	// Logging to see if arguments parsed correctly
	// fmt.Println(nodeNum)
	// fmt.Println(numberOfNodes)
	// fmt.Println(datasetName)

	logger = govec.InitGoVector(os.Args[1], os.Args[1])	

	//figure out ports of other clients
	myPort = strconv.Itoa(nodeNum + basePort)
	for i := 0; i < numberOfNodes; i++ {
		if strconv.Itoa(basePort+i) == myPort {
			continue
		}
		clusterPorts = append(clusterPorts, strconv.Itoa(basePort+i))
	}

	//initialize honest client and lock for updates

	client = Honest{id: nodeNum, blockUpdates: make([]Update, 0, 5)}

	// fmt.Println("ClientID:%d", client.id)	
	client.initializeData(datasetName, numberOfNodes)

	// verifier = false

	updateLock = sync.Mutex{}
	blockLock = sync.Mutex{}
	boolLock = sync.Mutex{}
	// fmt.Println(nodeNum)
	
	prepareForNextIteration()

	go messageListener(myPort)
	go messageSender(clusterPorts)


	//load data -- done
	// start listening for messages -- 

	// Clients irrespective need to Listen and send. This If should go inside the functions


	// if (verifier){		
	// 	// fmt.Printf("Am verifier\n")
	// 	updateLock.Lock() 
	// 	updates = []Update{} 
	// 	updateLock.Unlock()	
	// 	go messageListener(myPort)	
	
	// }else{
	
	// }

	select{}
}

func prepareForNextIteration() {
	
		
	// boolLock.Lock()
	// fmt.Println("IterationCHange: Lock Acquired")

	if(verifier){
		updateLock.Lock()
		client.flushUpdates(numberOfNodes)
		updateLock.Unlock()
	}
	iterationCount++
	verifier = amVerifier(client.id)	

	if(iterationCount>10){
		client.bc.PrintChain()
		os.Exit(1)
	}

	if(verifier){
		updateSent = true		
	}else{
		updateSent = false
	}

	// fmt.Println("IterationCHange: Lock Released")


	// boolLock.Unlock()




		

	


	// TODO: Run a VRF here.
	portsToConnect = make([]string, len(clusterPorts))

	copy(portsToConnect, clusterPorts)
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
	// fmt.Printf("Stepsize:%d\n",stepsize)

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

	// nodeData := dividedData[nodeID]
	filename :=  datasetName + strconv.Itoa(nodeID) + ".csv"
	file, err := os.Create(datasetPath + filename)
	check(err)
	nodeData.WriteCSV(bufio.NewWriter(file))
}

func messageListener(port string) {
	// Listen on port
	fmt.Printf("Listening on %s\n", port)

	myaddr, err := net.ResolveTCPAddr("tcp", myIP+port)
	if err != nil {
		fmt.Println("LISTEN ERROR")
	}

	ln, err := net.ListenTCP("tcp", myaddr)
	if err != nil {
		fmt.Println("Could not listen for messages")
	}

	// Reponse to messages
	for {
		fmt.Println("Waiting for a message...")

		conn, err := ln.Accept()
		if err != nil {
			fmt.Println("Could not accept connection on port")
		}

		fmt.Println("Got a ping.")
		go packetListener(conn)
		
	}
}


func packetListener(conn net.Conn) {


	inBuf := make([]byte, 2048)

	outBuf := make([]byte, 2048)

	// handleConnection (conn)

	n, err := conn.Read(inBuf)
	if err != nil {
		fmt.Printf("Got a reply read failure reading %d bytes.\n", n)
		conn.Close()
		os.Exit(1)
	}

	var message Message;
	logger.UnpackReceive("received message", inBuf[0:n], &message)

	switch message.Type {

		case "update":

			// fmt.Println(message.UpdateData)
			fmt.Printf("Got update message %d bytes, iteration %d\n", n, message.UpdateData.Iteration)

			updateLock.Lock() 

			for message.UpdateData.Iteration > iterationCount {
				// A crappy way to block until our iteration matches
				fmt.Printf("Blocking. Got update for %d, I am at %d\n", message.UpdateData.Iteration, iterationCount)
				time.Sleep(1000 * time.Millisecond)
			}

			
			numberOfUpdates := client.addBlockUpdate(message.UpdateData) 
			updateLock.Unlock()


			fmt.Println(numberOfUpdates) 
			if(numberOfUpdates == (numberOfNodes - 1)){
				// fmt.Println("here")
				blockToSend := client.createBlock(iterationCount)
				// fmt.Println(blockToSend)
				blockMsg := Message{Type: "block", Block: blockToSend}
				outBuf = logger.PrepareSend("Sending block to all other nodes", blockMsg)
				
				
				fmt.Printf("Sending block. Iteration: %d\n", blockMsg.Block.Data.Iteration)
				for _, port := range clusterPorts {

					conn, err := net.Dial("tcp", verifierIP+port)
					if err != nil {
						fmt.Printf("Could not connect to %s\n", port)
						// portsToRetry = append(portsToRetry, port)
						continue
					}
					n, err := conn.Write(outBuf)
					// fmt.Println(n)
					if err != nil {
						fmt.Println("Got a conn write failure writing %d bytes.", n)
						conn.Close()
						os.Exit(1)
					}						
				}
				// client.bc.PrintChain()
				prepareForNextIteration()
			}

		case "block":

			// fmt.Println(message.Block)			
			fmt.Printf("Got block message %d bytes, iteration %d\n", n, message.Block.Data.Iteration)

			// blockLock.Lock() 

			for message.Block.Data.Iteration > iterationCount {
				// A crappy way to block until our iteration matches
				fmt.Printf("Blocking. Got block for %d, I am at %d\n", message.Block.Data.Iteration, iterationCount)
				time.Sleep(1000 * time.Millisecond)
			}

			client.bc.AddBlockMsg(message.Block)			

			// blockLock.Unlock()

			// client.bc.PrintChain()		
			// fmt.Println("Iteration done")

			prepareForNextIteration()

	}



	// switch message.Type {

	// case "update":

	// 	fmt.Printf("Got update message %d bytes, iteration %d\n", n, message.UpdateData.Iteration)
	// 	updateLock.Lock() 
	// 	fmt.Println(message.UpdateData)
	// 	numberOfUpdates := client.addBlockUpdate(message.UpdateData) 
	// 	updateLock.Unlock()
	// 	fmt.Println(numberOfNodes -  1) // why the fuck is thois -1? I'll be back
	// 	if(numberOfUpdates == (numberOfNodes - 1)){
	// 		fmt.Println("here")
	// 		client.createBlock(iterationCount)
	// 		//send Block
	// 	}


	// Get a request. If the iteration count is correct, reply with your gradient.
	// case "request":


	// 	outMessage := Message{Type: "update", UpdateData: client.update}
	// 	outBuf = logger.PrepareSend("Sending update", outMessage)
	// 	fmt.Printf("Sending update %d\n", iterationCount)

	// CF: This whole section is confusing, I change it. But should it easy to integrate.
	// case "block":
	// 	fmt.Printf("Got block message %d bytes, iteration %d\n", n, message.Block.Data.Iteration)
	// 	// CF: You will need to do "first block wins" semantics here
	// 	for message.RequestData.Iteration > iterationCount {
	// 		// A crappy way to block until our iteration matches
	// 		fmt.Printf("Blocking. Got block for %d, I am at %d\n", message.RequestData.Iteration, iterationCount)
	// 		time.Sleep(1000 * time.Millisecond)
	// 	}

	// 	if message.Block.Data.Iteration == iterationCount {
	// 		if verifyBlock(message.Block) {
	// 			didReceiveBlock = true
	// 			receivedBlock = message.Block
	// 			bc.AppendBlock(&receivedBlock)
	// 		} else {
	// 			fmt.Printf("Bad block. Hash was incorrect\n")
	// 		}
	// 	} else {
	// 		fmt.Printf("Bad block. Got %d but iteration is on %d \n", message.RequestData.Iteration, iterationCount)
	// 	}

	// 	outMessage := Message{Type: "ack", AckData: Ack{Iteration: iterationCount}}
	// 	outBuf = logger.PrepareSend("Sending ack", outMessage)
	// 	fmt.Printf("Sending ack %d\n", iterationCount)

	// default:
	// 	fmt.Println("Unexpected message type")
	// }

	// n, err = conn.Write(outBuf)
	// fmt.Printf("Replying with a %d byte message.\n", n)
	// if err != nil {
	// 	fmt.Println("Got a conn write failure.")
	// 	conn.Close()
	// 	os.Exit(1)
	// }

	conn.Close()
}

func messageSender(ports []string) {
	
	outBuf := make([]byte, 2048)

	for {

		// fmt.Printf("Starting at iteration %d\n", iterationCount)
		// boolLock.Lock()

		if(verifier){

			// boolLock.Unlock()
			continue;
		}

		if(!updateSent){
			
			fmt.Printf("Computing Update\n")

			// blockLock.Lock()
			client.computeUpdate(iterationCount, datasetName)
			// blockLock.Unlock()

			portsToConnect = VRF(iterationCount)

			// CF: In future, portsToConnect will change every iteration, as the VRF is run.
			portsToRetry := []string{}
			for _, port := range portsToConnect {

				// First, establish a connection.
				conn, err := net.Dial("tcp", verifierIP+port)

				if err != nil {
					fmt.Printf("Could not connect to %s\n", port)
					portsToRetry = append(portsToRetry, port)
					continue
				}

				var msg Message

				msg.Type = "update"
				msg.UpdateData = client.update

				outBuf = logger.PrepareSend("Sending update to verifier", msg)

				n, err := conn.Write(outBuf)

				fmt.Printf("Update sent %d bytes, Iteration: %d\n " , n, msg.UpdateData.Iteration)

				// fmt.Println(n)
				if err != nil {
					fmt.Println("Got a conn write failure writing %d bytes.", n)
					conn.Close()
					os.Exit(1)
				}
				
			}
			updateSent = true		
			
		}
		// boolLock.Unlock()


			// fmt.Printf("Sent request %d byte, iteration %d\n", n, iterationCount)

			// // Get the response from the connection
			// inBuf := make([]byte, 1024)
			// n, err = conn.Read(inBuf)
			// if err != nil {
			// 	fmt.Printf("Got a reply read failure reading %d bytes.\n", n)
			// 	conn.Close()
			// 	os.Exit(1)
			// }

		// 	updateMessage := &Message{}
		// 	logger.UnpackReceive("received message", inBuf[0:n], &replyMessage)

		// 	fmt.Printf("Got update message %d bytes, iteration %d\n", n, replyMessage.UpdateData.Iteration)
		// 	updateLock.Lock()
		// 	updates = append(updates, replyMessage.UpdateData)
		// 	updateLock.Unlock()
		// 	fmt.Printf("Stored update %d\n", replyMessage.UpdateData.Iteration)

		// }

		// // Once received all updates start generating block
	// 	if len(updates) == len(clusterPorts)+1 {

	// 		block, err := generateBlock()
	// 		if err == nil {
	// 			fmt.Printf("Prev. hash: %x\n", block.PrevBlockHash)
	// 			fmt.Printf("Data: %s\n", block.Data.String())
	// 			fmt.Printf("Hash: %x\n", block.Hash)
	// 			ackCount := 0
	// 			portsToRetry2 := []string{}
	// 			for _, port := range portsToConnect {

	// 				// First, establish a connection.
	// 				conn, err := net.Dial("tcp", localHost+port)

	// 				if err != nil {
	// 					fmt.Printf("Could not connect to %s\n", port)
	// 					portsToRetry2 = append(portsToRetry2, port)
	// 					continue
	// 				}
	// 				var n int
	// 				// Send block
	// 				message := Message{Type: "block", Block: block}
	// 				outBuf := logger.PrepareSend("Sending block", message)
	// 				n, err = conn.Write(outBuf)
	// 				if err != nil {
	// 					fmt.Println("Got a conn write failure.")
	// 				}

	// 				// Get the response from the connection
	// 				inBuf := make([]byte, 1024)
	// 				n, err = conn.Read(inBuf)
	// 				if err != nil {
	// 					fmt.Printf("Got a reply read failure reading %d bytes.\n", n)
	// 					conn.Close()
	// 					os.Exit(1)
	// 				}

	// 				replyMessage := &Message{}
	// 				logger.UnpackReceive("received message", inBuf[0:n], &replyMessage)

	// 				fmt.Printf("Got update message %d bytes, iteration %d\n", n, replyMessage.AckData.Iteration)
	// 				ackCount++
	// 				fmt.Printf("Counted ack %d\n", replyMessage.AckData.Iteration)
	// 			}

	// 			if ackCount == len(clusterPorts)+1 {

	// 			} else {
	// 				portsToConnect = portsToRetry
	// 				time.Sleep(retryDuration * time.Millisecond)
	// 			}
	// 		} else {
	// 			// CF: I don't understand why this is here?
	// 			fmt.Printf("Prev. hash: %x\n", receivedBlock.PrevBlockHash)
	// 			fmt.Printf("Received Data: %s\n", receivedBlock.Data.String())
	// 			fmt.Printf("Hash: %x\n", receivedBlock.Hash)
	// 		}

	// 		prepareForNextIteration()
	// 		msg := "Updating iteration count to " + strconv.Itoa(iterationCount)
	// 		fmt.Println(msg)
	// 		logger.LogLocalEvent(msg)

	// 	} else {
	// 		portsToConnect = portsToRetry
	// 		time.Sleep(retryDuration * time.Millisecond)
	// 	}
	}
}

func VRF(iterationCount int) []string{

	// THIS WILL CHANGE AS THE VRF IMPLEMENTATION CHANGES
	verifiers := make([]string, 1)
	verifiers[0] = strconv.Itoa(basePort + (iterationCount % numberOfNodes))
	return verifiers	
}

func (msg Message) ToByte() []byte {

	var msgBytes bytes.Buffer 
	enc:= gob.NewEncoder(&msgBytes)       
    err := enc.Encode(msg)
    if err != nil {
        fmt.Println("encode error:", err)
    }

    // fmt.Println(blockDataBytes.Bytes())
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


