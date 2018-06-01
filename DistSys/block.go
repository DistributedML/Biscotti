package main

// import "fmt"

// Block storing updates, aggregation, and other metadata

import (
	"bytes"
	"crypto/sha256"
	"strconv"
	"time"
	// "github.com/sbinet/go-python"
)

type Block struct {
	Timestamp     int64
	Data          BlockData
	PrevBlockHash []byte
	Hash          []byte
}

func (b *Block) SetHash() {
	timestamp := []byte(strconv.FormatInt(b.Timestamp, 10))
	headers := bytes.Join([][]byte{b.PrevBlockHash, b.Data.ToByte(), timestamp}, []byte{})
	hash := sha256.Sum256(headers)
	b.Hash = hash[:]
}

func NewBlock(data BlockData, prevBlockHash []byte) *Block {
	block := &Block{time.Now().Unix(), data, prevBlockHash, []byte{}}
	block.SetHash()
	return block
}

func GenesisBlock(numFeatures int) *Block {

	genesisBlockData := BlockData{0, make([]float64, numFeatures), []Update{}} // create a globalWW with the appropriate number of features
	block := &Block{0, genesisBlockData, []byte{}, []byte{}}
	block.SetHash()
	return block

}
