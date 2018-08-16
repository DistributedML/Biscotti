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
func (myvrf *VRF) getNodes(stakeMap map[int]int, input []byte, size int, totalNodes int) ([]int, []byte, []byte) {

    vrfOutput, vrfProof := myvrf.sk.Prove(input)

    nodesMap := make(map[int]bool)
    lottery := []int{}
    verifiers := []int{}
    i := 0

    // Set up the lottery tickets
    for nodeid := 0; nodeid < totalNodes; nodeid++ {
        stake := stakeMap[nodeid]
        for i := 0; i < stake; i++ {
            lottery = append(lottery, nodeid)
        }
    }

    var winner int
    fmt.Println(lottery)

    for len(verifiers) < size {

        /*fmt.Println(input)
        fmt.Println(lottery)*/
        winnerIdx := (int(input[i]) * 256 + int(input[i+1])) % len(lottery)
        winner = lottery[winnerIdx]
        
        fmt.Print("Lottery winner is: ")
        fmt.Println(winnerIdx)

        _, exists := nodesMap[winner]
        if !exists{
            nodesMap[winner] = true
            verifiers = append(verifiers, winner)
        }
        
        i++
    }

    return verifiers, vrfOutput, vrfProof

}