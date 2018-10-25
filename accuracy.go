package mlmetrics

import (
	"sync"
)

// Accuracy is a basic classification metric. It measures how often the
// classifier makes the correct prediction. It is the ratio between the
// weight of correct predictions and the total weight of predictions.
type Accuracy struct {
	observed float64
	correct  float64
	mu       sync.RWMutex
}

// NewAccuracy inits a new metric.
func NewAccuracy() *Accuracy {
	return &Accuracy{}
}

// Reset resets state.
func (m *Accuracy) Reset() {
	m.mu.Lock()
	m.observed = 0
	m.correct = 0
	m.mu.Unlock()
}

// Observe records an observation of the actual vs the predicted category.
func (m *Accuracy) Observe(actual, predicted int) {
	m.ObserveWeight(actual, predicted, 1.0)
}

// ObserveWeight records an observation of the actual vs the predicted category with a given weight.
func (m *Accuracy) ObserveWeight(actual, predicted int, weight float64) {
	if !isValidCategory(actual) || !isValidCategory(predicted) || !isValidWeight(weight) {
		return
	}

	equal := predicted == actual

	m.mu.Lock()
	m.observed += weight
	if equal {
		m.correct += weight
	}
	m.mu.Unlock()
}

// TotalWeight returns the total weight observed.
func (m *Accuracy) TotalWeight() float64 {
	m.mu.RLock()
	observed := m.observed
	m.mu.RUnlock()
	return observed
}

// CorrectWeight returns the weight of correct observations.
func (m *Accuracy) CorrectWeight() float64 {
	m.mu.RLock()
	correct := m.correct
	m.mu.RUnlock()
	return correct
}

// Rate returns the rate of correct predictions.
func (m *Accuracy) Rate() float64 {
	m.mu.RLock()
	observed := m.observed
	correct := m.correct
	m.mu.RUnlock()

	if observed == 0 {
		return 0
	}
	return correct / observed
}
