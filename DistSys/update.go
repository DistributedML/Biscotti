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
	Iteration 	int
	Delta     	[]float64
	Commitment  []byte // can't be kyber.Point
	Accepted  	bool
}

func (update Update) String() string {

	suite := bn256.NewSuite()
	byteCommitment := update.Commitment

	pointCommitment := suite.G1().Point().Null()

	err := pointCommitment.UnmarshalBinary(byteCommitment)
	check(err)

	return fmt.Sprintf("{Iteration:" + strconv.Itoa(update.Iteration) + ", "  + "Commitment:" + pointCommitment.String() + ", "  + "Deltas:" + arrayToString(update.Delta, ",") + "}")

}

func arrayToString(a []float64, delim string) string {
	str := "[" + strings.Trim(strings.Replace(fmt.Sprint(a), " ", delim, -1), "[]") + "]"
	return str

}






