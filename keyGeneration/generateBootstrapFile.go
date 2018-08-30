package main

import (

	"fmt"
	"os"
	"bufio"
	"flag"
	"strconv"
	"bytes"

)

// Assumption: This script takes in the amount of nodes to be run at each machine in the hosts file alongwith the dimensions of delta w

var(

	fileReadPath = "../azure-deploy/tempHosts"
	peersFilePath = "../DistSys/peersfile.txt"
	commitKeyG1Path = "../DistSys/commitKeyG1.bin"
	commitKeyG2Path = "../DistSys/commitKeyG2.bin"
	pKeyG1Path = "../DistSys/pKeyG1.bin"
	sKeyPath = "../DistSys/sKey.bin"

	nodesInEachVM int
	dimensions int
	delimiter = "\n"

)

func main() {
	
	nodesInEachVMPtr := (flag.Int("n", 0 , "The total number of nodes in the network"))
	dimesionsPtr := (flag.Int("d", 0 , "The total number of nodes in the network"))

	flag.Parse()

	nodesInEachVM = *nodesInEachVMPtr 
	dimensions = *dimesionsPtr

	fmt.Println(nodesInEachVM)

	readFile, err := os.Open(fileReadPath)

	check(err)

	peersFile, er := os.Create(peersFilePath)
	commitKeyG1File, er := os.Create(commitKeyG1Path)
	commitKeyG2File, er := os.Create(commitKeyG2Path)
	pKeyG1File, er := os.Create(pKeyG1Path)
	sKeyFile, er := os.Create(sKeyPath)
	
	check(er)	

	defer readFile.Close()

	defer peersFile.Close()

	defer commitKeyG1File.Close()

	defer commitKeyG2File.Close()

	defer pKeyG1File.Close()

	defer sKeyFile.Close()

	scanner := bufio.NewScanner(readFile)

	hostindex := 0

	commitKey := PublicKey{}

	commitKey.GenerateKey(dimensions)

	var writeBufferG1 bytes.Buffer
	var writeBufferG2 bytes.Buffer


	for i := 0; i < len(commitKey.PKG1); i++ {

		_ , err =commitKey.PKG1[i].MarshalTo(&writeBufferG1)
		_ , err =commitKey.PKG2[i].MarshalTo(&writeBufferG2)

		check(err)
		// fmt.Print

		commitKeyG1File.Write(writeBufferG1.Bytes())
		commitKeyG1File.WriteString(delimiter)
		writeBufferG1.Reset()

		commitKeyG2File.Write(writeBufferG2.Bytes())
		commitKeyG2File.WriteString(delimiter)
		writeBufferG2.Reset()
	}

	var writeBufferPK bytes.Buffer
	var writeBufferSK bytes.Buffer


	for scanner.Scan() {
		
		thisLine := scanner.Text()
		fmt.Println(thisLine)	

		for i := hostindex; i < hostindex + nodesInEachVM; i++ {
			
			myPort := 8000 + i
			
			// adding IP:Port to file
			lineToWrite := thisLine + ":" + strconv.Itoa(myPort)+"\n"

			
			fmt.Println(lineToWrite)
			// generate public key and add that too with a comma
			
			thisClientKey := PublicKey{}

			secretKey := thisClientKey.GenerateClientKey()

			_ , err =thisClientKey.PKG1[0].MarshalTo(&writeBufferPK)
			check(err)

			_ , err =secretKey.MarshalTo(&writeBufferSK)
			check(err)

			pKeyG1File.Write(writeBufferPK.Bytes())
			pKeyG1File.WriteString(delimiter)
			writeBufferPK.Reset()

			sKeyFile.Write(writeBufferSK.Bytes())
			sKeyFile.WriteString(delimiter)
			writeBufferSK.Reset()

			

			_ , e := peersFile.WriteString(lineToWrite)

			check(e)
			peersFile.Sync()

		}

		hostindex = hostindex + nodesInEachVM

	}

}


func check(e error) {
    if e != nil {
        panic(e)
    }
}