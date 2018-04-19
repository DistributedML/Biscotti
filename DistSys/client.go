package main

// Client using the system
type Client interface {
	// Fetch all data needed for an aggregation
	sampleData()
	// Aggregate the sampledData
	computeUpdate() float64
	// Aggregate method to compute and verify blocks
	aggregate() float64
	// Verify block using given procedure
	verify(block Block) bool
}
