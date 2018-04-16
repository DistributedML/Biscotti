package main

import(
	"fmt"
	// "encoding/binary"
	  "encoding/gob"
	"bytes"
)

type BlockData struct {
	Iteration   int
	Aggregation float64
	Updates     []Update
}

//create nw BlockData

func NewBlockData(iteration int, aggregation float64) *BlockData {
	blockData := &BlockData{iteration, aggregation, []Update{}}
	// block.SetHash()
	return blockData
}

func (blockdata BlockData) String() string {
	return fmt.Sprintf("Iteration: %d, Aggregation: %f, Updates: %v",
		blockdata.Iteration, blockdata.Aggregation, blockdata.Updates)
}

//converts blockData to an array of bytes

func (blockdata BlockData) ToByte() []byte {

	var blockDataBytes bytes.Buffer        
    err := enc.Encode(blockdata)
    if err != nil {
        fmt.Println("encode error:", err)
    }

    fmt.Println(blockDataBytes.Bytes())
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









