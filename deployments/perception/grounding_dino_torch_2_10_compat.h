#pragma once

// Grounding DINO still passes Tensor::type() to AT_DISPATCH_* at the reviewed
// upstream revision.  PyTorch 2.10's header-only dispatcher accepts only its
// ScalarType enum.  Pre-include this header when building groundingdino._C so
// the official source checkout can remain unmodified and revision-verifiable.

#include <ATen/core/DeprecatedTypeProperties.h>
#include <torch/headeronly/core/Dispatch.h>

namespace detail {
inline torch::headeronly::ScalarType scalar_type(
    const at::DeprecatedTypeProperties& properties) {
  return static_cast<torch::headeronly::ScalarType>(properties.scalarType());
}
}  // namespace detail
