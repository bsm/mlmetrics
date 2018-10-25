package mlmetrics

import (
	"math"
	"sync"
)

// LogLoss, aka logistic loss or cross-entropy loss.
type LogLoss struct {
	epsilon float64
	logsum  float64
	weight  float64

	mu sync.RWMutex
}

// NewLogLoss inits a log-loss metric.
func NewLogLoss() *LogLoss {
	return NewLogLossWithEpsilon(0)
}

// NewLogLossWithEpsilon inits a log-loss metric with epsilon, a small increment to add to avoid
// taking a log of zero. Default: 1e-15.
func NewLogLossWithEpsilon(epsilon float64) *LogLoss {
	if epsilon <= 0 {
		epsilon = 1e-15
	}
	return &LogLoss{epsilon: epsilon}
}

// Reset resets state.
func (m *LogLoss) Reset() {
	m.mu.Lock()
	m.logsum = 0
	m.weight = 0
	m.mu.Unlock()
}

// Observe records the predicted probability of the actually observed value.
// Assuming the predictions were:
//   [dog: 0.2, cat: 0.5, fish: 0.3]
//   [dog: 0.8, cat: 0.1, fish: 0.1]
//   [dog: 0.6, cat: 0.1, fish: 0.4]
// And the actual observations were:
//   * cat
//   * dog
//   * fish
// Then the recorded values should be:
//   m.Observe(0.5)
//   m.Observe(0.8)
//   m.Observe(0.4)
func (m *LogLoss) Observe(prob float64) {
	m.ObserveWeight(prob, 1.0)
}

// ObserveWeight records an observation with a given weight.
func (m *LogLoss) ObserveWeight(prob float64, weight float64) {
	if !isValidProbability(prob) || !isValidWeight(weight) {
		return
	}

	if prob == 0 {
		prob += m.epsilon
	}

	m.mu.Lock()
	m.weight += weight
	m.logsum += weight * math.Log(prob)
	m.mu.Unlock()
}

// Score calculates the logarithmic loss.
func (m *LogLoss) Score() float64 {
	m.mu.RLock()
	logsum := m.logsum
	weight := m.weight
	m.mu.RUnlock()

	if weight > 0 {
		return -logsum / weight
	}
	return -math.Log(m.epsilon)
}
