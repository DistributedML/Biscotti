package main

import (
	"fmt"
	"strconv"
	"strings"
	"github.com/dedis/kyber/pairing/bn256"
	// "encoding/binary"
	// "bytes"
)

// Update - data object representing a single update
type Update struct {
	SourceID 		int
	Iteration 		int
	Delta     		[]float64
	Commitment  	[]byte // a commitment to delta: can't be kyber.Point
	Noise 			[]float64
	NoisedDelta 	[]float64
	Accepted  		bool
	SignatureList	[][]byte
}

func (update Update) String() string {

	suite := bn256.NewSuite()
	byteCommitment := update.Commitment

	pointCommitment := suite.G1().Point().Null()

	_ = pointCommitment.UnmarshalBinary(byteCommitment)

	return fmt.Sprintf("{Iteration:" + strconv.Itoa(update.Iteration) + ", "  + "Commitment:" + pointCommitment.String() + ", "  + "Deltas:" + arrayToString(update.Delta, ",") + "}")

}

func arrayToString(a []float64, delim string) string {
	str := "[" + strings.Trim(strings.Replace(fmt.Sprint(a), " ", delim, -1), "[]") + "]"
	return str

}