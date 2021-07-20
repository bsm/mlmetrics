package mlmetrics_test

import (
	"testing"

	. "github.com/bsm/ginkgo"
	. "github.com/bsm/gomega"
)

func TestSuite(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "mlmetrics")
}

// --------------------------------------------------------------------

func repeatInts(v int, repeat int) []int {
	nn := make([]int, repeat)
	for i := 0; i < len(nn); i++ {
		nn[i] = v
	}
	return nn
}

func intVector(slices ...[]int) (nn []int) {
	for _, s := range slices {
		nn = append(nn, s...)
	}
	return nn
}
