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
    alice := []byte("alice")
 
    aliceVRF := sk.Compute(alice)
    aliceVRFFromProof, aliceProof := sk.Prove(alice)

    fmt.Printf("pk:           %X\n", pk)
    fmt.Printf("sk:           %X\n", sk)
    fmt.Printf("alice(bytes): %X\n", alice)
    fmt.Printf("aliceVRF:     %X\n", aliceVRF)
    fmt.Printf("aliceProof:   %X\n", aliceProof)
    fmt.Printf("aliceVRFProof:   %X\n", aliceVRFFromProof)

    fmt.Print("Public Key Verification: ")
    fmt.Println(pk.Verify(alice, aliceVRF, aliceProof))
    
    fmt.Print("VRF Byte comparison: ")
    fmt.Println(bytes.Equal(aliceVRF, aliceVRFFromProof))

    fmt.Print("I choose the nodes: ")
    fmt.Println(getNodeSet(aliceVRF, 5))
    
    fmt.Print("The VRF proof gets the nodes: ")
    fmt.Println(getNodeSet(aliceVRFFromProof, 5))

}

// very inefficient helper function to get the verifier set
// TODO: include proof of stake for a weighted selection scheme
func getNodeSet(input []byte, size int) []int {

    nodesMap := make(map[int]bool)
    out := make([]int, 0)
    i := 0

    for len(nodesMap) < size {
        candidate := int(input[i]) % 10
        _, exists := nodesMap[candidate]
        if !exists{
            nodesMap[candidate] = true
            out = append(out, candidate)
        }
        i++
    }

    return out

}