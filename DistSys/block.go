package main

// import "fmt"

// Block storing updates, aggregation, and other metadata

import (
	"bytes"
	"crypto/sha256"
	"strconv"
	"time"
)

type Block struct {
	Timestamp     int64
	Data          BlockData
	PrevBlockHash []byte
	Hash          []byte
	StakeMap	  map[int]int
}


func (b *Block) SetHash() {
	timestamp := []byte(strconv.FormatInt(b.Timestamp, 10))	
	headers := bytes.Join([][]byte{b.PrevBlockHash, timestamp, b.Data.ToByte()}, []byte{})
	hash := sha256.Sum256(headers)
	b.Hash = hash[:]
}

func NewBlock(data BlockData, prevBlockHash []byte, stakeMap map[int]int) *Block {
	
	var blockTime int64

	if (len(data.Deltas) == 0) {
		blockTime = 0
	} else {
		blockTime = time.Now().Unix()
	}

	block := &Block{blockTime, data, prevBlockHash, []byte{}, stakeMap}
	block.SetHash()
	
	return block
}

func GenesisBlock(numFeatures int) *Block {

	genesisBlockData := BlockData{-1, make([]float64, numFeatures), []Update{}} // create a globalWW with the appropriate number of features
	block := &Block{0, genesisBlockData, []byte{}, []byte{}, map[int]int{}}
	block.SetHash()
	return block

}