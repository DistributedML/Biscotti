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
	converged		bool

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

	//Parsing arguments nodeIndex, numberOfNodes, datasetname

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
	
	
	// reading data and other housekeeping before the 

	client.initializeData(datasetName, numberOfNodes)

	converged = false

	updateLock = sync.Mutex{}
	blockLock = sync.Mutex{}
	boolLock = sync.Mutex{}
	
	prepareForNextIteration()

	// start listening for updates and sending them at the same time
	go messageListener(myPort)
	go messageSender(clusterPorts)


	select{}
}

func prepareForNextIteration() {
	
	// reinitialize shared stuff and check if you are the verifier going into the next iteration

	if(verifier){
		updateLock.Lock()
		client.flushUpdates(numberOfNodes)
		updateLock.Unlock()
	}
	iterationCount++
	verifier = amVerifier(client.id)

	// exit if converged

	if(converged){
		client.bc.PrintChain()
		os.Exit(1)
	}

	if(verifier){
		updateSent = true		
	}else{
		updateSent = false
	}

	portsToConnect = make([]string, len(clusterPorts))

	copy(portsToConnect, clusterPorts)
}

func getData(filePath string) dataframe.DataFrame{

	// read data into the dataframe from the given filepath

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

	// divide the dataset equally among the number of nodes

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

func messageListener(port string) {
	
	// Listen for messages

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

	// handle messages based on whether they are updates or blocks


	inBuf := make([]byte, 2048)

	outBuf := make([]byte, 2048)


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
				blockToSend := client.createBlock(iterationCount)
				blockMsg := Message{Type: "block", Block: blockToSend}
				outBuf = logger.PrepareSend("Sending block to all other nodes", blockMsg)
				
				
				fmt.Printf("Sending block. Iteration: %d\n", blockMsg.Block.Data.Iteration)
				for _, port := range clusterPorts {

					conn, err := net.Dial("tcp", verifierIP+port)
					if err != nil {
						fmt.Printf("Could not connect to %s\n", port)
						continue
					}
					n, err := conn.Write(outBuf)
					if err != nil {
						fmt.Println("Got a conn write failure writing %d bytes.", n)
						conn.Close()
						os.Exit(1)
					}						
				}

				converged = client.checkConvergence()
				
				prepareForNextIteration()
			}

		case "block":
			fmt.Printf("Got block message %d bytes, iteration %d\n", n, message.Block.Data.Iteration)


			for message.Block.Data.Iteration > iterationCount {
				// A crappy way to block until our iteration matches
				fmt.Printf("Blocking. Got block for %d, I am at %d\n", message.Block.Data.Iteration, iterationCount)
				time.Sleep(1000 * time.Millisecond)
			}

			client.bc.AddBlockMsg(message.Block)			


			converged = client.checkConvergence()

			prepareForNextIteration()

	}


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
	
	//Continous for loop that checks if an update to be sent

	outBuf := make([]byte, 2048)

	for {


		if(verifier){

			continue;
		}

		if(!updateSent){
			
			fmt.Printf("Computing Update\n")

			client.computeUpdate(iterationCount, datasetName)

			portsToConnect = VRF(iterationCount)

			portsToRetry := []string{}
			for _, port := range portsToConnect {

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

				if err != nil {
					fmt.Println("Got a conn write failure writing %d bytes.", n)
					conn.Close()
					os.Exit(1)
				}
				
			}
			updateSent = true		
			
		}

	}
}

func VRF(iterationCount int) []string{

	// THIS WILL CHANGE AS THE VRF IMPLEMENTATION CHANGES
	verifiers := make([]string, 1)
	verifiers[0] = strconv.Itoa(basePort + (iterationCount % numberOfNodes))
	return verifiers	
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


