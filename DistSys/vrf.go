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
// TODO: include proof of stake for a weighted selection scheme
func (myvrf *VRF) getNodes(input []byte, size int, total int) ([]int, []byte, []byte) {

    vrfOutput, vrfProof := myvrf.sk.Prove(input)

    nodesMap := make(map[int]bool)
    out := make([]int, 0)
    i := 0

    for len(nodesMap) < size {
        candidate := int(input[i]) % total
        _, exists := nodesMap[candidate]
        if !exists{
            nodesMap[candidate] = true
            out = append(out, candidate)
        }
        i++
    }

    return out, vrfOutput, vrfProof

}