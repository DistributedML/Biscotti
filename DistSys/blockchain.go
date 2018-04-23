package main

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