#pragma once

// Grounding DINO still passes Tensor::type() to AT_DISPATCH_* at the reviewed
// upstream revision. Recent PyTorch dispatchers accept only a ScalarType enum.
// Pre-include this header when building groundingdino._C so
// the official source checkout can remain unmodified and revision-verifiable.

#include <ATen/core/DeprecatedTypeProperties.h>

#if __has_include(<torch/headeronly/core/Dispatch.h>)
#include <torch/headeronly/core/Dispatch.h>

namespace detail {
inline torch::headeronly::ScalarType scalar_type(
    const at::DeprecatedTypeProperties& properties) {
  return static_cast<torch::headeronly::ScalarType>(properties.scalarType());
}
}  // namespace detail
#else
#include <ATen/Dispatch.h>

namespace detail {
inline at::ScalarType scalar_type(const at::DeprecatedTypeProperties& properties) {
  return properties.scalarType();
}
}  // namespace detail
#endif
