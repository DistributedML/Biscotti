package main

import (

	"fmt"
	"os"
	"bufio"
	"flag"
	"strconv"
	"bytes"
	"github.com/dedis/kyber"
	"github.com/dedis/kyber/pairing/bn256"
	"encoding/json"



)

// Assumption: This script takes in the amount of nodes to be run at each machine in the hosts file alongwith the dimensions of delta w

type PkeyG1 struct{

	Id 		int
	Pkey 	[]byte
	Skey 	[]byte

}

var(

	fileReadPath = "../azure-deploy/tempHosts"
	peersFilePath = "../DistSys/peersfile.txt"
	commitKeyPath = "../DistSys/commitKey.json"
	// commitKeyG2Path = "../DistSys/commitKeyG2.json"
	pKeyG1Path = "../DistSys/pKeyG1.json"
	// sKeyPath = "../DistSys/sKey.bin"

	nodesInEachVM int
	dimensions int
	delimiter = "\r\n"

)



func main() {
	
	suite := bn256.NewSuite()

	nodesInEachVMPtr := (flag.Int("n", 0 , "The total number of nodes in the network"))
	dimesionsPtr := (flag.Int("d", 0 , "The total number of nodes in the network"))

	flag.Parse()

	nodesInEachVM = *nodesInEachVMPtr 
	dimensions = *dimesionsPtr

	fmt.Println(nodesInEachVM)

	readFile, err := os.Open(fileReadPath)

	check(err)

	peersFile, er := os.Create(peersFilePath)
	commitKeyFile, er := os.Create(commitKeyPath)
	// commitKeyG2File, er := os.Create(commitKeyG2Path)
	pKeyG1File, er := os.Create(pKeyG1Path)
	// sKeyFile, er := os.Create(sKeyPath)

	pKeyG1Enc := json.NewEncoder(pKeyG1File)
	commitKeyEnc := json.NewEncoder(commitKeyFile)
	
	check(er)	

	defer readFile.Close()

	defer peersFile.Close()

	defer commitKeyFile.Close()

	// defer commitKeyG2File.Close()

	defer pKeyG1File.Close()

	// defer sKeyFile.Close()

	scanner := bufio.NewScanner(readFile)

	hostindex := 0

	commitKey := PublicKey{}

	commitKey.GenerateKey(dimensions)

	fmt.Println(commitKey)

	var writeBufferG1 bytes.Buffer


	for i := 0; i < dimensions; i++ {

		_ , err =commitKey.PKG1[i].MarshalTo(&writeBufferG1)
		thisPointG2Bytes , e := commitKey.PKG2[i].MarshalBinary()

		check(e)

		check(err)

		// fmt.Println(commitKey.PKG1[i])
		// fmt.Println(commitKey.PKG2[i])

		thisPointG2 := suite.G2().Point()

		err = thisPointG2.UnmarshalBinary(thisPointG2Bytes)

		check(err)

		// fmt.Println(thisPointG2)

		// fmt.Println("No error. MAGIC")

		// thisClientPKStruct := PkeyG1{Id: strconv.Itoa(i), PKey:writeBufferPK.Bytes(), SKey:writeBufferSK.Bytes()}
		thisCommitPKStruct := PkeyG1{Id: i, Pkey:writeBufferG1.Bytes(), Skey:thisPointG2Bytes}

		err = commitKeyEnc.Encode(&thisCommitPKStruct)

		check(err)

		writeBufferG1.Reset()

	}

	var writeBufferPK bytes.Buffer
	var writeBufferSK bytes.Buffer

	// pkeyMap := make(map[string]interface{})


	for scanner.Scan() {
		
		thisLine := scanner.Text()
		// fmt.Println(thisLine)	

		for i := hostindex; i < hostindex + nodesInEachVM; i++ {
			
			myPort := 8000 + i
			
			// adding IP:Port to file
			lineToWrite := thisLine + ":" + strconv.Itoa(myPort)+"\n"

			_ , e := peersFile.WriteString(lineToWrite)

			check(e)
			peersFile.Sync()


			
			// fmt.Println(lineToWrite)
			
			// generate public, secret key file
			
			thisClientKey := PublicKey{}

			secretKey := thisClientKey.GenerateClientKey()

			_ , err =thisClientKey.PKG1[0].MarshalTo(&writeBufferPK)
			check(err)

			_ , err =secretKey.MarshalTo(&writeBufferSK)
			check(err)

			// thisClientPKStruct := PkeyG1{Id: strconv.Itoa(i), PKey:writeBufferPK.Bytes(), SKey:writeBufferSK.Bytes()}
			thisClientPKStruct := PkeyG1{Id: i, Pkey:writeBufferPK.Bytes(), Skey:writeBufferSK.Bytes()}

			writeBufferPK.Reset()	

			writeBufferSK.Reset()		

			// fmt.Println(thisClientKey.PKG1[0])

			err = pKeyG1Enc.Encode(&thisClientPKStruct)

			check(err)

		}

		hostindex = hostindex + nodesInEachVM

	}

	fmt.Println("Done writing")

	extractPublicKeys()

	// pubKeyMap, sKey := extractKeys(0)

	// fmt.Println(pubKeyMap)
	// fmt.Println(len(pubKeyMap))
	// fmt.Println(sKey)

	commitmentKey := extractCommitmentKey(dimensions)

	fmt.Println(commitmentKey)

}


func check(e error) {
    if e != nil {
        panic(e)
    }
}

func extractPublicKeys() {

	suite := bn256.NewSuite()

	pKeyG1File, err := os.Open(pKeyG1Path)
	check(err)

	defer pKeyG1File.Close()

	scanner := bufio.NewScanner(pKeyG1File)

	for scanner.Scan() {

		thisKeyBytes := scanner.Bytes()
		
		thisKey := PkeyG1{}

		json.Unmarshal(thisKeyBytes, &thisKey)

		thisPoint := suite.G1().Point().Null()

		err := thisPoint.UnmarshalBinary(thisKey.Pkey)

		// fmt.Println(thisPoint)
		// fmt.Println(thisKey.Id)
		
		check(err)

	}	

}

func extractKeys(nodeNum int) (map[int]PublicKey, kyber.Scalar){
	
	pubKeyMap := make(map[int]PublicKey)

	suite := bn256.NewSuite()

	mySkey := suite.G1().Scalar().One()

	thisPoint := suite.G1().Point().Null()

	pKeyG1File, err := os.Open(pKeyG1Path)
	check(err)

	defer pKeyG1File.Close()

	scanner := bufio.NewScanner(pKeyG1File)

	for scanner.Scan() {

		thisKeyBytes := scanner.Bytes()
		
		thisKey := PkeyG1{}

		json.Unmarshal(thisKeyBytes, &thisKey)		

		err = thisPoint.UnmarshalBinary(thisKey.Pkey)

		check(err)

		thisPubKey := PublicKey{}

		thisPubKey.SetG1Key(thisPoint)

		pubKeyMap[thisKey.Id] = thisPubKey 	

		// Write Set Key function for this 

		// pubKeyMap[thisKey.Id].PKG1[0] = thisPoint

		// fmt.Println(thisPoint)
		

		if(thisKey.Id == nodeNum){

			mySkey.UnmarshalBinary(thisKey.Skey)			
		
		}


	}

	return pubKeyMap, mySkey	

}

func extractCommitmentKey(dimensions int) PublicKey {	

	suite := bn256.NewSuite()

	commitKey := PublicKey{PKG1:make([]kyber.Point, dimensions), PKG2:make([]kyber.Point, dimensions)}

	// commitKey.GenerateKey()


	commitKeyFile, err := os.Open(commitKeyPath)

	check(err)

	scanner := bufio.NewScanner(commitKeyFile)

	// index:=0

	for scanner.Scan() {
						
		thisKeyBytes := scanner.Bytes()
		
		thisKey := PkeyG1{}

		json.Unmarshal(thisKeyBytes, &thisKey)		

		thisPointG1 := suite.G1().Point()	

		err = thisPointG1.UnmarshalBinary(thisKey.Pkey)

		check(err)

		thisPointG2 := suite.G2().Point()

		// fmt.Println(len(thisKey))

		err = thisPointG2.UnmarshalBinary(thisKey.Skey)

		check(err)

		// fmt.Println(thisKey.Id)
		// fmt.Println(thisPointG1)
		// fmt.Println(thisPointG2)

		commitKey.PKG1[thisKey.Id] = thisPointG1.Clone()

		commitKey.PKG2[thisKey.Id] = thisPointG2.Clone()	


	}

	return commitKey	

}

