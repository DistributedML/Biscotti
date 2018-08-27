package main

import (
    "fmt"
    "github.com/coniks-sys/coniks-go/crypto/vrf"
)

type VRF struct {
    sk vrf.PrivateKey
    pk vrf.PublicKey
}

func (myvrf *VRF) init() {

    var err error
    myvrf.sk, err = vrf.GenerateKey(nil)
    if err != nil {
        fmt.Println("Error! Could not generate secret key")
    }

    myvrf.pk, _ = myvrf.sk.Public()

}

func (myvrf *VRF) getPublicKey() vrf.PublicKey {
    return myvrf.pk
}

func (myvrf *VRF) compute(input []byte) ([]byte, []byte) {
 
    // inputVRF := sk.Compute(input)
    // inputVRFFromProof, inputProof := sk.Prove(input)

    /*fmt.Printf("pk:           %X\n", pk)
    fmt.Printf("sk:           %X\n", sk)
    fmt.Printf("input(bytes): %X\n", input)
    fmt.Printf("inputVRF:     %X\n", inputVRF)
    fmt.Printf("inputProof:   %X\n", inputProof)
    fmt.Printf("inputVRFProof:   %X\n", inputVRFFromProof)

    fmt.Print("Public Key Verification: ")
    fmt.Println(pk.Verify(input, inputVRF, inputProof))
    
    fmt.Print("VRF Byte comparison: ")
    fmt.Println(bytes.Equal(inputVRF, inputVRFFromProof))

    fmt.Print("I choose the nodes: ")
    fmt.Println(getNodeSet(inputVRF, 5))
    
    fmt.Print("The VRF proof gets the nodes: ")
    fmt.Println(getNodeSet(inputVRFFromProof, 5)) */

    return myvrf.sk.Prove(input)

}

func (myvrf *VRF) verify(input []byte, theirPk vrf.PublicKey, inputVRF []byte, inputProof []byte) bool {
    return theirPk.Verify(input, inputVRF, inputProof)
} 

// very inefficient helper function to get the verifier set
// Based on stakeMap, nodes get lottery tickets proportional to their stake
func (myvrf *VRF) getNodes(stakeMap map[int]int, input []byte, size int, 
    totalNodes int) ([]int, []int, []int, []byte, []byte) {

    vrfOutput, vrfProof := myvrf.sk.Prove(input)

    lottery := []int{}

    vNodesMap := make(map[int]bool)
    verifiers := []int{}

    mNodesMap := make(map[int]bool)    
    miners := []int{}
    
    nNodesMap := make(map[int]bool)
    noisers := []int{}
    
    i := 0

    // Set up the lottery tickets
    for nodeid := 0; nodeid < totalNodes; nodeid++ {
        stake := stakeMap[nodeid]
        for i := 0; i < stake; i++ {
            lottery = append(lottery, nodeid)
        }
    }

    fmt.Println(input)
    fmt.Println(len(input))

    var winner int
    for len(verifiers) < size {

        /*fmt.Println(input)
        fmt.Println(lottery)*/
        winnerIdx := (int(input[i]) * 256 + int(input[i+1])) % len(lottery)
        winner = lottery[winnerIdx]
        
        outLog.Printf("Verifier lottery winner is %d at %d \n", winner, winnerIdx)

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

        outLog.Printf("Miner lottery winner is %d at %d \n", winner, winnerIdx)

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

        outLog.Printf("Noise lottery winner is %d at %d \n", winner, winnerIdx)

        _, exists := nNodesMap[winner]
        if !exists{
            nNodesMap[winner] = true
            noisers = append(noisers, winner)
        }
        
        i++
    }

    return verifiers, miners, noisers, vrfOutput, vrfProof
}