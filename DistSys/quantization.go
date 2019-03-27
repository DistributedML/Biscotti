package main

import "math"

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
