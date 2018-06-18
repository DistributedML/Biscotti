package main

import (
	"fmt"
)

type Blockchain struct {
	Blocks []*Block
}

func (bc *Blockchain) AddBlock(data BlockData) {
	prevBlock := bc.Blocks[len(bc.Blocks)-1]
	newBlock := NewBlock(data, prevBlock.Hash)
	bc.Blocks = append(bc.Blocks, newBlock)
}

func NewGenesisBlock(numFeatures int) *Block {
	return GenesisBlock(numFeatures)
}

func NewBlockchain(numFeatures int) *Blockchain {
	return &Blockchain{[]*Block{NewGenesisBlock(numFeatures)}}
}

func (bc *Blockchain) getLatestGradient() []float64 {

	prevBlock := bc.Blocks[len(bc.Blocks)-1]
	gradient := make([]float64, len(prevBlock.Data.GlobalW))
	copy(gradient, prevBlock.Data.GlobalW)
	return gradient
}

func (bc *Blockchain) PrintChain() {

	for _, block := range bc.Blocks {
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
	bc.Blocks = append(bc.Blocks, appendBlock)
}

func (bc *Blockchain) getBlock(iterationCount int) (*Block) {

	if((iterationCount + 1) <= len(bc.Blocks) - 1){
		return bc.Blocks[iterationCount+1]
	}else{
		return nil 
	}
	

}
