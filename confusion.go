package mlmetrics

import (
	"math"
	"sync"
)

// ConfusionMatrix can be used to visualize the performance of a binary
// classifier.
type ConfusionMatrix struct {
	mat resizableMatrix
	mu  sync.RWMutex
}

// NewConfusionMatrix inits a new ConfusionMatrix.
func NewConfusionMatrix() *ConfusionMatrix {
	return new(ConfusionMatrix)
}

// Reset resets the state.
func (m *ConfusionMatrix) Reset() {
	m.mu.Lock()
	m.mat = resizableMatrix{}
	m.mu.Unlock()
}

// Observe records an observation of the actual vs the predicted category.
func (m *ConfusionMatrix) Observe(actual, predicted int) {
	m.ObserveWeight(actual, predicted, 1.0)
}

// ObserveWeight records an observation of the actual vs the predicted category with a given weight.
func (m *ConfusionMatrix) ObserveWeight(actual, predicted int, weight float64) {
	if !isValidCategory(actual) || !isValidCategory(predicted) || !isValidWeight(weight) {
		return
	}

	m.mu.Lock()
	m.mat.Set(actual, predicted, m.mat.At(actual, predicted)+weight)
	m.mu.Unlock()
}

// Order returns the matrix order (number or rows/cols).
func (m *ConfusionMatrix) Order() int {
	m.mu.RLock()
	size := m.mat.size
	m.mu.RUnlock()

	return size
}

// TotalWeight returns the total weight observed (sum of the matrix).
func (m *ConfusionMatrix) TotalWeight() float64 {
	m.mu.RLock()
	sum := m.mat.Sum()
	m.mu.RUnlock()

	return sum
}

// Row returns the distribution of predicted weights for category x.
func (m *ConfusionMatrix) Row(x int) []float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if x >= m.mat.size {
		return nil
	}

	row := make([]float64, m.mat.size)
	copy(row, m.mat.Row(x))
	return row
}

// Column returns the distribution of actual weights for category x.
func (m *ConfusionMatrix) Column(x int) []float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if x >= m.mat.size {
		return nil
	}

	col := make([]float64, m.mat.size)
	for i := 0; i < m.mat.size; i++ {
		col[i] = m.mat.At(i, x)
	}
	return col
}

// Accuracy returns the overall accuracy rate.
func (m *ConfusionMatrix) Accuracy() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	sum := m.mat.Sum()
	if sum == 0.0 {
		return 0.0
	}

	var pos float64
	for i := 0; i < m.mat.size; i++ {
		pos += m.mat.At(i, i)
	}
	return pos / sum
}

// Precision calculates the positive predictive value for category x.
func (m *ConfusionMatrix) Precision(x int) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	sum := m.mat.ColSum(x)
	if sum == 0.0 {
		return 0.0
	}
	return m.mat.At(x, x) / sum
}

// Sensitivity calculates the recall (aka 'hit rate') for category x.
func (m *ConfusionMatrix) Sensitivity(x int) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	sum := m.mat.RowSum(x)
	if sum == 0.0 {
		return 0.0
	}
	return m.mat.At(x, x) / sum
}

// F1 calculates the F1 score for category x, the harmonic mean of precision and sensitivity.
func (m *ConfusionMatrix) F1(x int) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	csm := m.mat.ColSum(x)
	if csm == 0 {
		return 0
	}

	rsm := m.mat.RowSum(x)
	if rsm == 0 {
		return 0
	}

	pos := m.mat.At(x, x)
	precision := pos / csm
	sensitivity := pos / rsm
	return 2 * precision * sensitivity / (precision + sensitivity)
}

// Kappa represents the Cohen's Kappa, a statistic which measures inter-rater agreement for qualitative
// (categorical) items. It is generally thought to be a more robust measure than simple percent agreement
// calculation, as κ takes into account the possibility of the agreement occurring by chance.
// https://en.wikipedia.org/wiki/Cohen%27s_kappa
func (m *ConfusionMatrix) Kappa() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	sum := m.mat.Sum()
	if sum == 0.0 {
		return 0.0
	}

	var obs, exp float64
	for i := 0; i < m.mat.size; i++ {
		obs += m.mat.At(i, i)
		exp += m.mat.RowSum(i) * m.mat.ColSum(i) / sum
	}
	if div := sum - exp; div != 0 {
		return (obs - exp) / div
	}
	return 1.0
}

// Matthews is a correlation coefficient used as a measure of the quality of binary
// and multiclass classifications. It takes into account true and false positives
// and negatives and is generally regarded as a balanced measure which can be
// used even if the classes are of very different sizes. The MCC is in essence
// a correlation coefficient value between -1 and +1. A coefficient of +1 represents
// a perfect prediction, 0 an average random prediction and -1 an inverse prediction.
// The statistic is also known as the phi coefficient. [source: Wikipedia]
func (m *ConfusionMatrix) Matthews() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	sum := m.mat.Sum()
	if sum == 0.0 {
		return 0.0
	}

	var exp, cf1, cf2, cf3 float64
	for i := 0; i < m.mat.size; i++ {
		rsum := m.mat.RowSum(i)
		csum := m.mat.ColSum(i)

		exp += m.mat.At(i, i)
		cf1 += rsum * csum
		cf2 += rsum * rsum
		cf3 += csum * csum
	}

	sum2 := sum * sum
	if pdt := (sum2 - cf2) * (sum2 - cf3); pdt != 0 {
		return ((exp * sum) - cf1) / math.Sqrt(pdt)
	}
	return 0
}

type resizableMatrix struct {
	size int
	data []float64
}

// Set sets the field at (i, j) to v
func (m *resizableMatrix) Set(i, j int, v float64) {
	m.resize(maxInt(i+1, j+1))
	m.data[i*m.size+j] = v
}

// At returns the value at (i, j)
func (m *resizableMatrix) At(i, j int) float64 {
	if i < m.size && j < m.size {
		return m.data[i*m.size+j]
	}
	return 0
}

// Row returns the slice of row at i.
func (m *resizableMatrix) Row(i int) []float64 {
	if i >= 0 && i < m.size {
		offset := i * m.size
		return m.data[offset : offset+m.size]
	}
	return nil
}

// RowSum returns the sum of values in row (i).
func (m *resizableMatrix) RowSum(i int) (sum float64) {
	if i >= 0 && i < m.size {
		offset := (i * m.size)
		for k := offset; k < offset+m.size; k++ {
			sum += m.data[k]
		}
	}
	return
}

// RowSum returns the sum of values in col (j).
func (m *resizableMatrix) ColSum(j int) (sum float64) {
	if j >= 0 && j < m.size {
		for k := j; k < len(m.data); k += m.size {
			sum += m.data[k]
		}
	}
	return
}

// Sum calculates the sum of all cells.
func (m *resizableMatrix) Sum() float64 {
	sum := 0.0
	for _, v := range m.data {
		sum += v
	}
	return sum
}

func (m *resizableMatrix) resize(n int) {
	if n <= m.size {
		return
	}

	data := make([]float64, n*n)
	for row := 0; row < m.size; row++ {
		offset := row * m.size
		copy(data[row*n:], m.data[offset:offset+m.size])
	}
	m.size = n
	m.data = data
}
