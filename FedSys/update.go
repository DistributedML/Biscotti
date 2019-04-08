package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Update - data object representing a single update
type Update struct {
	SourceID 		int
	Iteration 		int
	Delta     		QuantizedWeights
}

func (update Update) String() string {
	return fmt.Sprintf("{Iteration:" + strconv.Itoa(update.Iteration) + ", " + "Deltas:" + arrayToStringUint8(update.Delta.Weights, ",") + "}")
}

func arrayToString(a []float64, delim string) string {
	str := "[" + strings.Trim(strings.Replace(fmt.Sprint(a), " ", delim, -1), "[]") + "]"
	return str

}

func arrayToStringUint8(a []uint8, delim string) string {
	str := "[" + strings.Trim(strings.Replace(fmt.Sprint(a), " ", delim, -1), "[]") + "]"
	return str

}






