package main

import (
	"fmt"
	"strconv"
	"strings"
	// "encoding/binary"
	// "bytes"
)

// Update - data object representing a single update
type Update struct {
	Iteration 	int
	Delta     	[]float64
	Noise 		[]float64
	NoisedDelta []float64
	Accepted  	bool
}

func (update Update) String() string {
	return fmt.Sprintf("{Iteration:" + strconv.Itoa(update.Iteration) + ", " + "Deltas:" + arrayToString(update.Delta, ",") + "}")
}

func arrayToString(a []float64, delim string) string {
	str := "[" + strings.Trim(strings.Replace(fmt.Sprint(a), " ", delim, -1), "[]") + "]"
	return str

}


