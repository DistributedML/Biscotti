package main

import (
    "bytes"
    "fmt"
    "github.com/coniks-sys/coniks-go/crypto/vrf"
    "strconv"
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

    stakeMap := make(map[string]int)
    for i := 0; i < 10; i++ {
        stakeMap[strconv.Itoa(i)] = i * 10
    }

    results := getNodeSet(stakeMap, aliceVRF, 3)
    fmt.Print("I choose the nodes: ")
    fmt.Println(results)

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
// TODO: include proof of stake for a weighted selection scheme
func getNodeSet(stakeMap map[string]int, input []byte, size int) []string {

    lottery := []string{}
    verifiers := []string{}

    for node, stake := range stakeMap {
        for i := 0; i < stake; i++ {
            lottery = append(lottery, node)
        }
    }

    fmt.Println(lottery)

    nodesMap := make(map[string]bool)
    i := 0

    var winner string

    for len(verifiers) < size {

        winnerIdx := (int(input[i]) * 256 + int(input[i+1])) % len(lottery)
        
        fmt.Print("Lottery winner is: ")
        fmt.Print(winnerIdx)
        fmt.Print(" out of ")
        fmt.Println(len(lottery))

        winner = lottery[winnerIdx]
        
        fmt.Println(winner)

        _, exists := nodesMap[winner]
        if !exists{
            nodesMap[winner] = true
            verifiers = append(verifiers, winner)
        }
        
        i++
    }

    return verifiers

}