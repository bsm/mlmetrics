package mlmetrics

import (
	"math"
	"sync"
)

// Regression is a basic regression evaluator
type Regression struct {
	weight float64 // total weight observed
	sum    float64 // sum of all values

	resSum   float64 // residual sum
	resSum2  float64 // residual sum of squares
	logSum2  float64 // logarithmic residual sum of squares
	totSum2  float64 // total sum of squares
	maxDelta float64 // maximum error delta

	mu sync.RWMutex
}

// NewRegression inits a new metric.
func NewRegression() *Regression {
	return &Regression{}
}

// Reset resets state.
func (m *Regression) Reset() {
	m.mu.Lock()
	m.weight = 0
	m.sum = 0
	m.resSum = 0
	m.resSum2 = 0
	m.logSum2 = 0
	m.totSum2 = 0
	m.maxDelta = 0
	m.mu.Unlock()
}

// Observe records an observation of the actual vs the predicted value.
func (m *Regression) Observe(actual, predicted float64) {
	m.ObserveWeight(actual, predicted, 1.0)
}

// ObserveWeight records an observation of the actual vs the predicted value with a given weight.
func (m *Regression) ObserveWeight(actual, predicted, weight float64) {
	if !isValidNumeric(actual) || !isValidNumeric(predicted) || !isValidWeight(weight) {
		return
	}

	residual := math.Abs(actual - predicted)
	logres := math.Abs(math.Log1p(actual) - math.Log1p(predicted))

	m.mu.Lock()
	defer m.mu.Unlock()

	if residual > m.maxDelta {
		m.maxDelta = residual
	}
	if m.weight != 0 {
		delta := actual - m.sum/m.weight
		m.totSum2 += delta * delta * weight
	}

	m.resSum += residual * weight
	m.resSum2 += residual * residual * weight
	m.logSum2 += logres * logres * weight

	m.sum += actual * weight
	m.weight += weight
}

// TotalWeight returns the total weight observed.
func (m *Regression) TotalWeight() float64 {
	m.mu.RLock()
	weight := m.weight
	m.mu.RUnlock()
	return weight
}

// MaxError returns the maximum observed error delta.
func (m *Regression) MaxError() float64 {
	m.mu.RLock()
	maxDelta := m.maxDelta
	m.mu.RUnlock()

	return maxDelta
}

// Mean returns the mean actual value observed.
func (m *Regression) Mean() float64 {
	m.mu.RLock()
	weight := m.weight
	sum := m.sum
	m.mu.RUnlock()

	if weight > 0 {
		return sum / weight
	}
	return 0.0
}

// MAE calculates the mean absolute error.
func (m *Regression) MAE() float64 {
	m.mu.RLock()
	weight := m.weight
	resSum := m.resSum
	m.mu.RUnlock()

	if weight > 0 {
		return resSum / weight
	}
	return 0.0
}

// MSE calculates the mean squared error.
func (m *Regression) MSE() float64 {
	m.mu.RLock()
	weight := m.weight
	resSum2 := m.resSum2
	m.mu.RUnlock()

	if weight > 0 {
		return resSum2 / weight
	}
	return 0.0
}

// MSLE calculates the mean squared logarithmic error loss.
func (m *Regression) MSLE() float64 {
	m.mu.RLock()
	weight := m.weight
	logSum2 := m.logSum2
	m.mu.RUnlock()

	if weight > 0 {
		return logSum2 / weight
	}
	return 0.0
}

// RMSE calculates the root mean squared error.
func (m *Regression) RMSE() float64 {
	return math.Sqrt(m.MSE())
}

// RMSLE calculates the root mean squared logarithmic error loss.
func (m *Regression) RMSLE() float64 {
	return math.Sqrt(m.MSLE())
}

// R2 calculates the R² coefficient of determination.
func (m *Regression) R2() float64 {
	m.mu.RLock()
	resSum2 := m.resSum2
	totSum2 := m.totSum2
	m.mu.RUnlock()

	if totSum2 > 0 {
		return 1 - resSum2/totSum2
	}
	return 0.0
}
