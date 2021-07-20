package mlmetrics_test

import (
	"fmt"
	"math"

	. "github.com/bsm/ginkgo"
	. "github.com/bsm/gomega"
	"github.com/bsm/mlmetrics"
)

var _ = Describe("Regression", func() {
	var subject *mlmetrics.Regression

	BeforeEach(func() {
		subject = mlmetrics.NewRegression()
		subject.Observe(26, 25)
		subject.Observe(20, 25)
		subject.Observe(24, 22)
		subject.Observe(21, 23)
		subject.Observe(23, 24)
		subject.Observe(25, 29)
		subject.Observe(27, 28)
		subject.ObserveWeight(28, 26, 2.0)
		subject.Observe(29, 30)
		subject.Observe(22, 18)
	})

	It("should calculate basic stats", func() {
		Expect(subject.TotalWeight()).To(Equal(11.0))
		Expect(subject.MaxError()).To(Equal(5.0))
		Expect(subject.Mean()).To(BeNumerically("~", 24.818, 0.001))
	})

	It("should mean errors", func() {
		Expect(subject.MAE()).To(BeNumerically("~", 2.273, 0.001))
		Expect(subject.MSE()).To(BeNumerically("~", 7.000, 0.001))
		Expect(subject.RMSE()).To(BeNumerically("~", 2.646, 0.001))
		Expect(subject.MSLE()).To(BeNumerically("~", 0.012, 0.001))
		Expect(subject.RMSLE()).To(BeNumerically("~", 0.110, 0.001))
	})

	It("should calculate R²", func() {
		Expect(subject.R2()).To(BeNumerically("~", 0.390, 0.01))

		subject.ObserveWeight(28, 28, 2.0)
		Expect(subject.R2()).To(BeNumerically("~", 0.477, 0.001))
	})

	It("should handle blanks", func() {
		subject.Reset()
		Expect(subject.TotalWeight()).To(Equal(0.0))
		Expect(subject.MaxError()).To(Equal(0.0))
		Expect(subject.Mean()).To(Equal(0.0))
		Expect(subject.MAE()).To(Equal(0.0))
		Expect(subject.MSE()).To(Equal(0.0))
		Expect(subject.RMSE()).To(Equal(0.0))
		Expect(subject.MSLE()).To(Equal(0.0))
		Expect(subject.RMSLE()).To(Equal(0.0))
	})

	It("should handle negative values", func() {
		subject.Observe(-28, -27)

		Expect(subject.MaxError()).To(Equal(5.0))
		Expect(subject.Mean()).To(BeNumerically("~", 20.417, 0.001))
		Expect(subject.MAE()).To(BeNumerically("~", 2.167, 0.001))
		Expect(subject.MSE()).To(BeNumerically("~", 6.500, 0.001))
		Expect(subject.RMSE()).To(BeNumerically("~", 2.550, 0.001))
		Expect(math.IsNaN(subject.MSLE())).To(BeTrue())
		Expect(math.IsNaN(subject.RMSLE())).To(BeTrue())
	})
})

func ExampleRegression() {
	yTrue := []float64{26, 20, 24, 21, 23, 25, 27}
	yPred := []float64{25, 25, 22, 23, 24, 29, 28}

	metric := mlmetrics.NewRegression()
	for i := range yTrue {
		metric.Observe(yTrue[i], yPred[i])
	}

	// print score
	fmt.Printf("count : %.0f\n", metric.TotalWeight())
	fmt.Printf("mae   : %.3f\n", metric.MAE())
	fmt.Printf("mse   : %.3f\n", metric.MSE())
	fmt.Printf("rmse  : %.3f\n", metric.RMSE())
	fmt.Printf("msle  : %.3f\n", metric.MSLE())
	fmt.Printf("rmsle : %.3f\n", metric.RMSLE())
	fmt.Printf("r2    : %.3f\n", metric.R2())

	// Output:
	// count : 7
	// mae   : 2.286
	// mse   : 7.429
	// rmse  : 2.726
	// msle  : 0.012
	// rmsle : 0.110
	// r2    : 0.162
}
