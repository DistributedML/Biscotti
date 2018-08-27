package main

import (
    "bytes"
    "fmt"
    "github.com/coniks-sys/coniks-go/crypto/vrf"
)

func main() {

    sk, err := vrf.GenerateKey(nil)
    if err != nil {
        fmt.Println("Error...")
    }

    pk, _ := sk.Public()
    alice := []byte("theLatestBlock")
 
    aliceVRF := sk.Compute(alice)
    aliceVRFFromProof, aliceProof := sk.Prove(alice)

/*    fmt.Printf("pk:           %X\n", pk)
    fmt.Printf("sk:           %X\n", sk)
    fmt.Printf("alice(bytes): %X\n", alice)
    fmt.Printf("aliceVRF:     %X\n", aliceVRF)
    fmt.Printf("aliceProof:   %X\n", aliceProof)
    fmt.Printf("aliceVRFProof:   %X\n", aliceVRFFromProof)*/

    fmt.Print("Public Key Verification: ")
    fmt.Println(pk.Verify(alice, aliceVRF, aliceProof))
    
    fmt.Print("VRF Byte comparison: ")
    fmt.Println(bytes.Equal(aliceVRF, aliceVRFFromProof))

    stakeMap := make(map[int]int)
    for i := 0; i < 20; i++ {
        stakeMap[i] = i * 10
    }

    verif, miners, noisers := getNodeSet(stakeMap, aliceVRF, 3)
    fmt.Print("Verifier nodes: ")
    fmt.Println(verif)

    fmt.Print("Miner nodes: ")
    fmt.Println(miners)

    fmt.Print("Noise nodes: ")
    fmt.Println(noisers)

    verif, miners, noisers = getNodeSet(stakeMap, aliceVRF, 3)
    fmt.Print("Verifier nodes: ")
    fmt.Println(verif)

    fmt.Print("Miner nodes: ")
    fmt.Println(miners)

    fmt.Print("Noise nodes: ")
    fmt.Println(noisers)

    /*
    
    fmt.Print("The VRF proof gets the nodes: ")
    fmt.Println(getNodeSet(aliceVRFFromProof, 5))

    for i := 0; i < 10; i++ {
        vrfValue, _ := sk.Prove(alice)
        alice = append(alice, 12)
        fmt.Println(getNodeSet(vrfValue, 5))
    }*/

}

// very inefficient helper function to get the verifier set
// returns verifiers, miners, noise providers
func getNodeSet(stakeMap map[int]int, input []byte, 
    size int) ([]int, []int, []int) {

    lottery := []int{}
    verifiers := []int{}
    miners := []int{}
    noisers := []int{}

    // Set up the lottery tickets
    for nodeid := 0; nodeid < len(stakeMap); nodeid++ {
        stake := stakeMap[nodeid]
        for i := 0; i < stake; i++ {
            lottery = append(lottery, nodeid)
        }
    }

    // fmt.Println(lottery)

    vNodesMap := make(map[int]bool)
    nNodesMap := make(map[int]bool)
    mNodesMap := make(map[int]bool)
    i := 0

    var winner int

    for len(verifiers) < size {

        winnerIdx := (int(input[i]) * 256 + int(input[i+1])) % len(lottery)
        
        fmt.Printf("Lottery winner is: %d \n", winnerIdx)

        winner = lottery[winnerIdx]
        
        // fmt.Println(winner)

        _, exists := vNodesMap[winner]
        if !exists{
            vNodesMap[winner] = true
            verifiers = append(verifiers, winner)
        }
        
        i++
    }

    for len(miners) < size {

        winnerIdx := (int(input[i]) * 256 + int(input[i+1])) % len(lottery)
        winner = lottery[winnerIdx]

        fmt.Printf("Lottery winner is: %d \n", winnerIdx)

        _, exists := mNodesMap[winner]
        if !exists{
            mNodesMap[winner] = true
            miners = append(miners, winner)
        }
        
        i++
    }

    for len(noisers) < size {

        winnerIdx := (int(input[i]) * 256 + int(input[i+1])) % len(lottery)
        winner = lottery[winnerIdx]

        fmt.Printf("Lottery winner is: %d \n", winnerIdx)

        _, exists := nNodesMap[winner]
        if !exists{
            nNodesMap[winner] = true
            noisers = append(noisers, winner)
        }
        
        i++
    }

    return verifiers, miners, noisers

}