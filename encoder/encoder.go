package encoder

import (
	ts "github.com/sugarme/gotch/tensor"
)

// Encoder is encoder interface for a image segmentation model.
type Encoder interface {
	ForwardAll(x *ts.Tensor, train bool) []*ts.Tensor
}
