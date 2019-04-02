package main

import (
	"fmt"
	// "encoding/binary"
	"bytes"
	"encoding/gob"
)

type BlockData struct {
	Iteration int
	GlobalW   QuantizedWeights
	Deltas    []Update
}

//create nw BlockData

func NewBlockData(iteration int, globalW QuantizedWeights, deltas []Update) *BlockData {
	blockData := &BlockData{iteration, globalW, deltas}
	// block.SetHash()
	return blockData
}

func (blockdata BlockData) String() string {
	return fmt.Sprintf("Iteration: %d, GlobalW: %s, Min: %s, Max: %s, deltas: %s",
		blockdata.Iteration, arrayToStringUint8(blockdata.GlobalW.Weights, ","), blockdata.GlobalW.Min,
		blockdata.GlobalW.Max, arrayToStringUpdate(blockdata.Deltas, ","))
}

//converts blockData to an array of bytes

func (blockdata BlockData) ToByte() []byte {

	var blockDataBytes bytes.Buffer
	enc := gob.NewEncoder(&blockDataBytes)
	err := enc.Encode(blockdata)
	if err != nil {
		fmt.Println("encode error:", err)
	}

	// fmt.Println(blockDataBytes.Bytes())
	return blockDataBytes.Bytes()

	// How to decode this thing. I will leave it here for future ref.

	// var q Q
	// err = dec.Decode(&q)
	// if err != nil {
	// Decode (receive) the value.
	//     log.Fatal("decode error:", err)
	// }
	// fmt.Printf("%q: {%d,%d}\n", q.Name, *q.X, *q.Y)

}

func arrayToStringUpdate(a []Update, delim string) string {

	updates := "["
	numUpdates := len(a)
	for i := 0; i < numUpdates; i++ {
		updates += a[i].String()
		if i != numUpdates-1 {
			updates += " " + delim
		}
	}
	updates += "]"
	// return strings.Trim(strings.Replace(fmt.Sprint(a.String), " ", delim, -1), "[]")
	return updates
}

// A more difficult version of byte conversion. Doesn't work
// but I dont want to throw it away

// func (blockdata BlockData) ToByte() []byte {
// 	var blockDataBytes []byte

// 	iterationBytes:= make([]byte,4)
// 	binary.LittleEndian.Putint32(iterationBytes, blockdata.Iteration)
// 	fmt.Println(iterationBytes)
// 	// err := binary.Write(iterationBytes, binary.LittleEndian, blockdata.Iteration)
// 	// if err != nil {
// 	// 	fmt.Println("Iteration:%d", blockdata.Iteration)
// 	// 	fmt.Println("got here 1")
//  //        fmt.Println("binary.Write failed:", err)
//  //    }
// 	blockDataBytes = append(blockDataBytes, iterationBytes.Bytes()...)

// 	aggregationBytes:= new(bytes.Buffer)
// 	err = binary.Write(aggregationBytes, binary.LittleEndian, blockdata.Aggregation)
// 	if err != nil {
// 		fmt.Println("got here 2")
//         fmt.Println("binary.Write failed:", err)
//     }
// 	blockDataBytes = append(blockDataBytes, aggregationBytes.Bytes()...)

// 	var updateBytes[]byte
// 	for _, update := range blockdata.Updates {
// 		updateBytes = append(updateBytes, update.ToByte()...)
// 	}

// 	blockDataBytes = append(blockDataBytes,updateBytes...)
// 	return blockDataBytes
// }
