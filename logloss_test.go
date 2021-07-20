package mlmetrics_test

import (
	"fmt"
	"math/rand"
	"testing"

	. "github.com/bsm/ginkgo"
	. "github.com/bsm/gomega"
	"github.com/bsm/mlmetrics"
)

var _ = Describe("LogLoss", func() {
	var subject *mlmetrics.LogLoss

	BeforeEach(func() {
		subject = mlmetrics.NewLogLoss()
	})

	It("should calculate score", func() {
		subject.Observe(0.5)
		subject.Observe(0.1)
		subject.Observe(0.01)
		subject.Observe(0.1)
		subject.Observe(0.25)
		subject.Observe(0.999)
		Expect(subject.Score()).To(BeNumerically("~", 1.882, 0.001))
	})

	It("should calculate score (variable)", func() {
		subject.Observe(0.8)
		subject.Observe(0.9)
		subject.Observe(0.1)
		subject.Observe(0.6)
		Expect(subject.Score()).To(BeNumerically("~", 0.785, 0.001))

		subject.Observe(0.0)
		Expect(subject.Score()).To(BeNumerically("~", 7.536, 0.001))

		subject.Observe(0.99)
		Expect(subject.Score()).To(BeNumerically("~", 6.282, 0.001))
	})

	It("should calculate on empty", func() {
		Expect(subject.Score()).To(BeNumerically("~", 34.539, 0.001))
	})

	It("should calculate perfect match", func() {
		subject.ObserveWeight(1.0, 10)
		Expect(subject.Score()).To(BeNumerically("~", 0.0, 0.001))

		subject.ObserveWeight(1.0, 10)
		Expect(subject.Score()).To(BeNumerically("~", 0.0, 0.001))
	})

	It("should calculate perfect failure", func() {
		subject.ObserveWeight(0.0, 10)
		Expect(subject.Score()).To(BeNumerically("~", 34.539, 0.001))

		subject.ObserveWeight(0.0, 10)
		Expect(subject.Score()).To(BeNumerically("~", 34.539, 0.001))
	})
})

func ExampleLogLoss() {
	// assuming the following three predictions
	predictions := []map[string]float64{
		{"dog": 0.2, "cat": 0.5, "fish": 0.3},
		{"dog": 0.8, "cat": 0.1, "fish": 0.1},
		{"dog": 0.6, "cat": 0.1, "fish": 0.4},
	}

	// create metric, feed with probabilities of actual observations
	metric := mlmetrics.NewLogLoss()
	for i, actual := range []string{"cat", "dog", "fish"} {
		probability := predictions[i][actual]
		metric.Observe(probability)
	}

	// print score
	fmt.Printf("log-loss : %.3f\n", metric.Score())

	// Output:
	// log-loss : 0.611
}

func BenchmarkLogLoss(b *testing.B) {
	rn := rand.New(rand.NewSource(10))
	ll := mlmetrics.NewLogLoss()

	for i := 0; i < 1000; i++ {
		ll.Observe(0.5 + rn.Float64()/2)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if v := ll.Score(); int(v*1000) != 301 {
			b.Fatalf("expected result to be 0.301 but was %v", v)
		}
	}
}
