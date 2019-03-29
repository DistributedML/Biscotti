package main

import "math"

type QuantizedWeights struct {
	Weights []uint8
	Min float64
	Max float64
}

func findMinAndMax(weights []float64) (float64, float64) {
	min := math.Inf(1)
	max := math.Inf(-1)
	for _, w := range weights {
		if w > max {
			max = w
		}
		if w < min {
			min = w
		}
	}
	return min, max
}

func quantizeWeights(weights []float64) QuantizedWeights {
	numOfIntervals := math.Pow(2, 8) - 1
	min, max := findMinAndMax(weights)
	quantizedWeights := make([]uint8, len(weights))
	for i, weight := range weights {
		quantizedWeight := quantizeWeight(weight, numOfIntervals, min, max)
		quantizedWeights[i] = quantizedWeight
	}
	return QuantizedWeights{Weights: quantizedWeights, Min: min, Max: max}
}

func quantizeWeight(weight float64, numOfIntervals float64, min float64, max float64) uint8 {
	diff := weight - min
	divisor := (max - min) / numOfIntervals
	quantizedRepresentation := int(diff / divisor)
	return uint8(quantizedRepresentation)
}

func dequantizeWeights(quantizedWeights QuantizedWeights) []float64 {
	weights := make([]float64, len(quantizedWeights.Weights))
	for i, quantizedWeight := range quantizedWeights.Weights {
		weights[i] = dequantizeWeight(quantizedWeight, quantizedWeights.Min, quantizedWeights.Max)
	}
	return weights
}

func dequantizeWeight(weight uint8, min float64, max float64) float64 {
	numOfIntervals := math.Pow(2, 8) - 1
	diff := max - min
	return float64(weight)*(diff / numOfIntervals) + min
}
