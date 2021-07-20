# ML metrics

[![GoDoc](https://godoc.org/github.com/bsm/mlmetrics?status.svg)](https://godoc.org/github.com/bsm/mlmetrics)
[![Test](https://github.com/bsm/openmetrics/actions/workflows/test.yml/badge.svg)](https://github.com/bsm/openmetrics/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Common metrics for evaluation of machine learning models.

Goals:

* Fast!
* Thread-safe
* Support for online evaluation

## Supported Metrics

Classification:

* [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision)
* [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
* [F1 Score](https://en.wikipedia.org/wiki/F1_score)
* [Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
* [Matthews](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
* [LogLoss](https://en.wikipedia.org/wiki/Loss_functions_for_classification)
* [Precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision)
* [Sensitivity](https://en.wikipedia.org/wiki/Sensitivity_(test))

Regression:

* [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error)
* [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error)
* [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
* [Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
* [R²](https://en.wikipedia.org/wiki/Coefficient_of_determination)

## Documentation

Documentation and example are available via godoc at http://godoc.org/github.com/bsm/mlmetrics

## Example

```go
package main

import (
	"github.com/bsm/mlmetrics"
)

func main() {{ "ExampleConfusionMatrix" | code }}
```
