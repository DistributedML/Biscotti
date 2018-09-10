package main

import (
    // "bytes"
    // "encoding/gob"
    "fmt"
    // "log"
    "github.com/dedis/kyber/pairing/bn256"
    "github.com/dedis/kyber"
)

func main() {
    // Initialize the encoder and decoder.  Normally enc and dec would be
    // bound to network connections and the encoder and decoder would
    // run in different processes.
	// gob.Register(MinerPart{})

    suite := bn256.NewSuite()
    // var network bytes.Buffer        // Stand-in for a network connection
    // enc := gob.NewEncoder(&network) // Will write to network.
    // dec := gob.NewDecoder(&network) // Will read from network.
    // Encode (send) the value.

    commit := (suite.G1().Point().Mul(suite.G1().Scalar().One(),nil))
    commit2 := suite.G1().Point().Mul(suite.G1().Scalar().SetInt64(4),commit)
    iteration := 0
    nodeId := 0

    polynomial := []int64{0,0,0,0,0,0}
    commitment := commit
    share1 := Share{X:10, Y:20}
    share2 := Share{X:12, Y:20}
    secrets := []Share{share1, share2}
    witnesses :=  []kyber.Point{commit, commit2}

    polyPart := PolynomialPart{Polynomial:polynomial, Commitment:commitment, Secrets:secrets, Witnesses:witnesses}

    polyMap := make(map[int]PolynomialPart)

    polyMap[0] = polyPart
 
    minerPart := MinerPart{CommitmentUpdate: commit, Iteration:iteration, NodeID: nodeId, PolyMap: polyMap}

    fmt.Println(minerPart)

    minerRPC := converttoRPC(minerPart)

    fmt.Println(minerRPC)

    minerPartConv := converttoMinerPart(minerRPC)

    fmt.Println(minerPartConv)


    
    // err := enc.Encode(MinerPart{CommitmentUpdate: commit, Iteration:iteration, NodeID: nodeId})
    // if err != nil {
    //     log.Fatal("encode error:", err)
    // }
    // // // Decode (receive) the value.
    // var q MinerPart
    // err = dec.Decode(&q)
    // if err != nil {
    //     log.Fatal("decode error:", err)
    // }

    // fmt.Println(q)
}