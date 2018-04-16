package main

// import "fmt"

// Block storing updates, aggregation, and other metadata

import(
	"strconv"
	"bytes"
	"time"
	"crypto/sha256"


)
type Block struct {
	Timestamp     int64
	data          BlockData
	PrevBlockHash []byte
	Hash          []byte
}

func (b *Block) SetHash() {
	timestamp := []byte(strconv.FormatInt(b.Timestamp, 10))
	headers := bytes.Join([][]byte{b.PrevBlockHash, b.data.ToByte(), timestamp}, []byte{})	
	hash := sha256.Sum256(headers)
	b.Hash = hash[:]
}

func NewBlock(data BlockData,  prevBlockHash []byte) *Block {
	block := &Block{time.Now().Unix(), data, prevBlockHash, []byte{}}
	block.SetHash()
	return block
}

func GenesisBlock() *Block{
	genesisBlockData := BlockData{-1, 0.0,[]Update{} }	
	block := &Block{time.Now().Unix(), genesisBlockData, []byte{},[]byte{}}
	block.SetHash()
	return block

}

