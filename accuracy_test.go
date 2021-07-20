package mlmetrics_test

import (
	. "github.com/bsm/ginkgo"
	. "github.com/bsm/gomega"
	"github.com/bsm/mlmetrics"
)

var _ = Describe("Accuracy", func() {
	var subject *mlmetrics.Accuracy

	BeforeEach(func() {
		subject = mlmetrics.NewAccuracy()
		subject.Observe(1, 1)
		subject.Observe(1, 1)
		subject.Observe(1, 0)
		subject.Observe(0, 0)
		subject.Observe(0, 0)
		subject.Observe(0, 1)
		subject.Observe(1, 1)
		subject.ObserveWeight(1, 1, 1.0)
		subject.ObserveWeight(1, 1, 2.0)
		subject.ObserveWeight(0, 0, 1.0)
		subject.ObserveWeight(0, 1, 1.0)
	})

	It("should calculate stats", func() {
		Expect(subject.CorrectWeight()).To(Equal(9.0))
		Expect(subject.TotalWeight()).To(Equal(12.0))
		Expect(subject.Rate()).To(Equal(0.75))
	})

	It("should ignore non-categories", func() {
		subject.Observe(-1, 1)
		subject.Observe(1, -1)
		subject.Observe(-1, -1)
		Expect(subject.TotalWeight()).To(Equal(12.0))
	})
})
