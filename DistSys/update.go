package main

import(
	"fmt"
	// "encoding/binary"
	// "bytes"
)

// Update - data object representing a single update
type Update struct {
	Iteration int
	Value     float64
}

func (update Update) String() string {
	return fmt.Sprintf("(I: %d, V: %f)", update.Iteration, update.Value)
}

// func (update Update) ToByte() []byte {
// 	var updateBytes []byte

// 	iterationBytes:= new(bytes.Buffer) 	
// 	err := binary.Write(iterationBytes, binary.LittleEndian, update.Iteration)	
// 	if err != nil {
// 		fmt.Println("Iteration:%d", update.Iteration)
// 		fmt.Println("got here 3")
//         fmt.Println("binary.Write failed:", err)
//     }
// 	updateBytes = append(updateBytes, iterationBytes.Bytes()...)
	
// 	valueBytes:= new(bytes.Buffer) 	
// 	err = binary.Write(valueBytes, binary.LittleEndian, update.Value)	
// 	if err != nil {
// 		fmt.Println("got here 4")
//         fmt.Println("binary.Write failed:", err)
//     }
// 	updateBytes = append(updateBytes, iterationBytes.Bytes()...)		

// 	return updateBytes

// } 
