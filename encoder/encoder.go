package encoder

import (
	"github.com/sugarme/gotch/ts"
)

// Encoder is encoder interface for a image segmentation model.
type Encoder interface {
	ForwardAll(x *ts.Tensor, train bool) []*ts.Tensor
}
