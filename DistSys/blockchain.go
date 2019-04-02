package main

import (
	"fmt"
	"os"
)

type Blockchain struct {
	Blocks []*Block
}

func (bc *Blockchain) AddBlock(data BlockData, stakeMap map[int]int) {
	prevBlock := bc.Blocks[len(bc.Blocks)-1]
	newBlock := NewBlock(data, prevBlock.Hash, stakeMap)
	bc.Blocks = append(bc.Blocks, newBlock)
}

func NewGenesisBlock(numFeatures int) *Block {
	return GenesisBlock(numFeatures)
}

func NewBlockchain(numFeatures int) *Blockchain {
	return &Blockchain{[]*Block{NewGenesisBlock(numFeatures)}}
}

func (bc *Blockchain) getLatestBlock() *Block {
	return bc.Blocks[len(bc.Blocks)-1]
}


func (bc *Blockchain) getLatestGradient() []float64 {
	prevBlock := bc.Blocks[len(bc.Blocks)-1]
	gradient := make([]float64, len(prevBlock.Data.GlobalW.Weights))
	prevBlockDequantizedWeights := dequantizeWeights(prevBlock.Data.GlobalW)
	copy(gradient, prevBlockDequantizedWeights)
	return gradient
}

func (bc *Blockchain) getLatestBlockModel() QuantizedWeights {
	return bc.Blocks[len(bc.Blocks)-1].Data.GlobalW
}

func (bc *Blockchain) PrintChain() {

	for _, block := range bc.Blocks {
		fmt.Printf("Prev. hash: %x\n", block.PrevBlockHash)
		fmt.Printf("Data: %s\n", block.Data.String())
		fmt.Printf("Hash: %x\n", block.Hash)
		// fmt.Printf("Stake: %v\n", block.StakeMap)
		fmt.Println()
	}

}

// Get hash of the last block
func (bc *Blockchain) getLatestBlockHash() []byte {
	outLog.Printf("Latest Block Hash is: %x", bc.Blocks[len(bc.Blocks)-1].Hash)
	return bc.Blocks[len(bc.Blocks)-1].Hash
}

func (bc *Blockchain) verifyBlock(block Block) bool {

	return true
}

func (bc *Blockchain) AddBlockMsg(newBlock Block) {

	appendBlock := &Block{Timestamp: newBlock.Timestamp, 
		Data: newBlock.Data, 
		PrevBlockHash: newBlock.PrevBlockHash, 
		Hash: newBlock.Hash, 
		StakeMap: newBlock.StakeMap}
		
	bc.Blocks = append(bc.Blocks, appendBlock)
}

func (bc *Blockchain) getBlock(iterationCount int) (*Block) {

	if (len(bc.Blocks) >= (iterationCount + 2)) {
		
		outLog.Printf("Returning a block")
		if(bc.Blocks[iterationCount+1].Data.Iteration != iterationCount){

			outLog.Printf("Something's fishy. Blocks for multiple iterations have been appended")
			bc.PrintChain()
			os.Exit(1)
		}
		return bc.Blocks[iterationCount+1]
	
	} else {
	
		outLog.Printf("Returning nil")
		return nil 
	}
	
}
