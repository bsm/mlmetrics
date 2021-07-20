package mlmetrics_test

import (
	"fmt"
	"testing"

	. "github.com/bsm/ginkgo"
	. "github.com/bsm/gomega"
	"github.com/bsm/mlmetrics"
)

var _ = Describe("ConfusionMatrix", func() {
	var subject *mlmetrics.ConfusionMatrix

	BeforeEach(func() {
		subject = mlmetrics.NewConfusionMatrix()
	})

	It("should calculate weights", func() {
		y1 := []int{0, 0, 1, 0, 0, 1, 1, 1}
		y2 := []int{1, 0, 1, 0, 0, 0, 0, 1}
		for i := range y1 {
			subject.Observe(y1[i], y2[i])
		}

		Expect(subject.Order()).To(Equal(2))
		Expect(subject.TotalWeight()).To(Equal(8.0))

		Expect(subject.Row(0)).To(Equal([]float64{3, 1}))
		Expect(subject.Row(1)).To(Equal([]float64{2, 2}))
		Expect(subject.Row(2)).To(BeNil())

		Expect(subject.Column(0)).To(Equal([]float64{3, 2}))
		Expect(subject.Column(1)).To(Equal([]float64{1, 2}))
		Expect(subject.Column(2)).To(BeNil())
	})

	It("should reset", func() {
		subject.ObserveWeight(0, 0, 40)
		subject.ObserveWeight(0, 1, 5)
		subject.ObserveWeight(1, 0, 5)
		subject.ObserveWeight(1, 1, 50)
		Expect(subject.Order()).To(Equal(2))
		Expect(subject.TotalWeight()).To(Equal(100.0))

		subject.Reset()
		Expect(subject.Order()).To(Equal(0))
		Expect(subject.TotalWeight()).To(Equal(0.0))
		Expect(subject.Row(0)).To(BeNil())
	})

	It("should calculate Precision", func() {
		y1 := intVector(repeatInts(0, 46), repeatInts(1, 44), repeatInts(2, 10))
		y2 := intVector(repeatInts(0, 52), repeatInts(1, 32), repeatInts(2, 16))
		for i := range y1 {
			subject.Observe(y1[i], y2[i])
		}
		Expect(subject.Precision(0)).To(BeNumerically("~", 0.884, 0.001))
		Expect(subject.Precision(1)).To(BeNumerically("~", 1.000, 0.001))
		Expect(subject.Precision(2)).To(BeNumerically("~", 0.625, 0.001))
		Expect(subject.Precision(3)).To(Equal(0.0))
	})

	It("should calculate Sensitivity", func() {
		y1 := intVector(repeatInts(0, 46), repeatInts(1, 44), repeatInts(2, 10))
		y2 := intVector(repeatInts(0, 52), repeatInts(1, 32), repeatInts(2, 16))
		for i := range y1 {
			subject.Observe(y1[i], y2[i])
		}
		Expect(subject.Sensitivity(0)).To(BeNumerically("~", 1.000, 0.001))
		Expect(subject.Sensitivity(1)).To(BeNumerically("~", 0.727, 0.001))
		Expect(subject.Sensitivity(2)).To(BeNumerically("~", 1.000, 0.001))
		Expect(subject.Sensitivity(3)).To(Equal(0.0))
	})

	It("should calculate F1 score", func() {
		y1 := intVector(repeatInts(0, 46), repeatInts(1, 44), repeatInts(2, 10))
		y2 := intVector(repeatInts(0, 52), repeatInts(1, 32), repeatInts(2, 16))
		for i := range y1 {
			subject.Observe(y1[i], y2[i])
		}
		Expect(subject.F1(0)).To(BeNumerically("~", 0.939, 0.001))
		Expect(subject.F1(1)).To(BeNumerically("~", 0.842, 0.001))
		Expect(subject.F1(2)).To(BeNumerically("~", 0.769, 0.001))
		Expect(subject.F1(3)).To(Equal(0.0))
	})

	It("should calculate Accuracy", func() {
		y1 := intVector(repeatInts(0, 46), repeatInts(1, 44), repeatInts(2, 10))
		y2 := intVector(repeatInts(0, 52), repeatInts(1, 32), repeatInts(2, 16))
		for i := range y1 {
			subject.Observe(y1[i], y2[i])
		}
		Expect(subject.Accuracy()).To(BeNumerically("~", 0.880, 0.001))
	})

	Describe("Kappa", func() {
		// These label vectors reproduce the contingency matrix from Artstein and
		// Poesio (2008), Table 1.
		It("should calculate scores (binary)", func() {
			y1 := intVector(repeatInts(0, 40), repeatInts(1, 60))
			y2 := intVector(repeatInts(0, 20), repeatInts(1, 20), repeatInts(0, 10), repeatInts(1, 50))
			for i := range y1 {
				subject.Observe(y1[i], y2[i])
			}
			Expect(subject.Kappa()).To(BeNumerically("~", 0.348, 0.001))
		})

		// Multiclass example: Artstein and Poesio, Table 4.
		It("should calculate scores (multiclass)", func() {
			y1 := intVector(repeatInts(0, 46), repeatInts(1, 44), repeatInts(2, 10))
			y2 := intVector(repeatInts(0, 52), repeatInts(1, 32), repeatInts(2, 16))
			for i := range y1 {
				subject.Observe(y1[i], y2[i])
			}
			Expect(subject.Kappa()).To(BeNumerically("~", 0.801, 0.001))
		})

		It("should calculate weighted scores", func() {
			subject.ObserveWeight(0, 0, 22)
			subject.ObserveWeight(0, 1, 7)
			subject.ObserveWeight(1, 0, 9)
			subject.ObserveWeight(1, 1, 13)
			Expect(subject.Kappa()).To(BeNumerically("~", 0.353, 0.001))
		})

		It("should calculate on empty", func() {
			Expect(subject.Kappa()).To(Equal(0.0))

			subject.Observe(0, 1)
			Expect(subject.Kappa()).To(Equal(0.0))
		})

		It("should calculate full agreement", func() {
			subject.Observe(0, 0)
			subject.Observe(1, 1)
			Expect(subject.Kappa()).To(Equal(1.0))
		})

		It("should calculate no agreement", func() {
			subject.Observe(0, 1)
			subject.Observe(1, 0)
			Expect(subject.Kappa()).To(Equal(-1.0))
		})
	})

	Describe("Matthews", func() {
		It("should calculate scores (binary)", func() {
			y1 := intVector(repeatInts(0, 40), repeatInts(1, 60))
			y2 := intVector(repeatInts(0, 20), repeatInts(1, 20), repeatInts(0, 10), repeatInts(1, 50))
			for i := range y1 {
				subject.Observe(y1[i], y2[i])
			}
			Expect(subject.Matthews()).To(BeNumerically("~", 0.356, 0.001))
		})

		It("should calculate scores (multiclass)", func() {
			y1 := intVector(repeatInts(0, 46), repeatInts(1, 44), repeatInts(2, 10))
			y2 := intVector(repeatInts(0, 52), repeatInts(1, 32), repeatInts(2, 16))
			for i := range y1 {
				subject.Observe(y1[i], y2[i])
			}
			Expect(subject.Matthews()).To(BeNumerically("~", 0.816, 0.001))
		})

		It("should calculate full agreement (binary)", func() {
			y1 := []int{1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1}
			y2 := []int{1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1}
			for i := range y1 {
				subject.Observe(y1[i], y2[i])
			}
			Expect(subject.Matthews()).To(Equal(1.0))
		})

		It("should calculate full agreement (multiclass)", func() {
			y1 := []int{0, 0, 1, 2}
			y2 := []int{0, 0, 1, 2}
			for i := range y1 {
				subject.Observe(y1[i], y2[i])
			}
			Expect(subject.Matthews()).To(Equal(1.0))
		})

		It("should calculate no agreement (binary)", func() {
			y1 := []int{1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1}
			y2 := []int{1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1}
			for i := range y1 {
				subject.Observe(y1[i], y2[i])
			}
			Expect(subject.Matthews()).To(Equal(0.0))
		})

		It("should calculate no agreement (multiclass)", func() {
			y1 := []int{0, 0, 1, 1, 2, 2}
			y2 := []int{2, 2, 0, 0, 1, 1}
			for i := range y1 {
				subject.Observe(y1[i], y2[i])
			}
			Expect(subject.Matthews()).To(BeNumerically("~", -0.5, 0.0001))
		})

		It("should calculate no correlation (multiclass)", func() {
			y1 := []int{0, 1, 2, 0, 1, 2, 0, 1, 2}
			y2 := []int{1, 1, 1, 2, 2, 2, 0, 0, 0}
			for i := range y1 {
				subject.Observe(y1[i], y2[i])
			}
			Expect(subject.Matthews()).To(BeNumerically("~", 0.0, 0.0001))
		})

		It("should calculate no agreement (zero variance)", func() {
			y1 := []int{0, 1, 2}
			y2 := []int{3, 3, 3}
			for i := range y1 {
				subject.Observe(y1[i], y2[i])
			}
			Expect(subject.Matthews()).To(Equal(0.0))
		})
	})
})

func ExampleConfusionMatrix() {
	yTrue := []int{2, 0, 2, 2, 0, 1}
	yPred := []int{0, 0, 2, 2, 0, 2}

	mat := mlmetrics.NewConfusionMatrix()
	for i := range yTrue {
		mat.Observe(yTrue[i], yPred[i])
	}

	// print matrix
	for i := 0; i < mat.Order(); i++ {
		fmt.Println(mat.Row(i))
	}

	// print metrics
	fmt.Println()
	fmt.Printf("accuracy : %.3f\n", mat.Accuracy())
	fmt.Printf("kappa    : %.3f\n", mat.Kappa())
	fmt.Printf("matthews : %.3f\n", mat.Matthews())

	// Output:
	// [2 0 0]
	// [0 0 1]
	// [1 0 2]
	//
	// accuracy : 0.667
	// kappa    : 0.429
	// matthews : 0.452
}

func BenchmarkConfusionMatrix_Kappa(b *testing.B) {
	cm := mlmetrics.NewConfusionMatrix()
	y1 := intVector(repeatInts(0, 40), repeatInts(1, 60))
	y2 := intVector(repeatInts(0, 20), repeatInts(1, 20), repeatInts(0, 10), repeatInts(1, 50))
	for i := range y1 {
		cm.Observe(y1[i], y2[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if v := cm.Kappa(); int(v*1000) != 347 {
			b.Fatalf("expected result to be 0.347 but was %v", v)
		}
	}
}

func BenchmarkConfusionMatrix_Matthews(b *testing.B) {
	cm := mlmetrics.NewConfusionMatrix()
	y1 := intVector(repeatInts(0, 40), repeatInts(1, 60))
	y2 := intVector(repeatInts(0, 20), repeatInts(1, 20), repeatInts(0, 10), repeatInts(1, 50))
	for i := range y1 {
		cm.Observe(y1[i], y2[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if v := cm.Matthews(); int(v*1000) != 356 {
			b.Fatalf("expected result to be 0.356 but was %v", v)
		}
	}
}
