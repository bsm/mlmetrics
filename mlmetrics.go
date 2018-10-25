package mlmetrics

import "math"

func maxInt(n, m int) int {
	if n > m {
		return n
	}
	return m
}

func isValidProbability(p float64) bool { return p >= 0 && p <= 1 }
func isValidWeight(w float64) bool      { return w > 0 }
func isValidCategory(x int) bool        { return x > -1 }
func isValidNumeric(v float64) bool     { return !math.IsNaN(v) }
