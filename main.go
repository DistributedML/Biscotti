package main

import(
	"fmt"
)

func main() {
	bc := NewBlockchain()

	bData1 := BlockData{1, 3.0,[]Update{Update{1,3.0}} }
	bData2 := BlockData{2, 9.0,[]Update{Update{1,3.0},Update{2,6.0}} }

	bc.AddBlock(bData1)
	bc.AddBlock(bData2)

	for _, block := range bc.blocks {
		fmt.Printf("Prev. hash: %x\n", block.PrevBlockHash)
		fmt.Printf("Data: %s\n", block.data.String())
		fmt.Printf("Hash: %x\n", block.Hash)
		fmt.Println()
	}
}

