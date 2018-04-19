package main

import (
	"math/rand"
)

const (
	samples         = 10 // L
	sampleDeviation = 0.1
)

// Honest Client
type Honest struct {
	id        int
	localData []float64
	update    Update
}

func (honest *Honest) sampleData() {
	// Sample data, use id as mean
	honest.localData = make([]float64, samples)
	for i := range honest.localData {
		honest.localData[i] = rand.NormFloat64()*sampleDeviation + float64(honest.id)
	}
}

func (honest *Honest) computeUpdate(iterationCount int) {
	sum := 0.0
	for _, data := range honest.localData {
		sum += data
	}
	honest.update = Update{iterationCount, sum / float64(samples)}
}

func (honest *Honest) aggregateUpdates(updates []Update) float64 {
	sum := 0.0
	for _, update := range updates {
		sum += update.Value
	}
	return sum / float64(len(updates))
}
