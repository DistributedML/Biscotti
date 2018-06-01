package main

import (
	"fmt"
)

type Blockchain struct {
	blocks []*Block
}

func (bc *Blockchain) AddBlock(data BlockData) {
	prevBlock := bc.blocks[len(bc.blocks)-1]
	newBlock := NewBlock(data, prevBlock.Hash)
	bc.blocks = append(bc.blocks, newBlock)
}

func NewGenesisBlock(numFeatures int) *Block {
	return GenesisBlock(numFeatures)
}

func NewBlockchain(numFeatures int) *Blockchain {
	return &Blockchain{[]*Block{NewGenesisBlock(numFeatures)}}
}

func (bc *Blockchain) getLatestGradient() []float64 {

	prevBlock := bc.blocks[len(bc.blocks)-1]
	gradient := make([]float64, len(prevBlock.Data.GlobalW))
	copy(gradient, prevBlock.Data.GlobalW)
	return gradient
}

func (bc *Blockchain) PrintChain() {

	for _, block := range bc.blocks {
		fmt.Printf("Prev. hash: %x\n", block.PrevBlockHash)
		fmt.Printf("Data: %s\n", block.Data.String())
		fmt.Printf("Hash: %x\n", block.Hash)
		fmt.Println()
	}

}

func (bc *Blockchain) verifyBlock(block Block) bool {

	return true
}

func (bc *Blockchain) AddBlockMsg(newBlock Block) {

	appendBlock := &Block{Timestamp: newBlock.Timestamp, Data: newBlock.Data, PrevBlockHash: newBlock.PrevBlockHash, Hash: newBlock.Hash}
	bc.blocks = append(bc.blocks, appendBlock)
}
