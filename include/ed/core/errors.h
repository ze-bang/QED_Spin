// =============================================================================
// include/ed/core/errors.h
//
// The library's error types: what a caller can catch and act on. The Python
// bindings translate each into the qed.errors class of the same name, which also
// derives from the builtin a caller would have caught before
// (python/qed/errors.py):
//
//   InvalidRequest     the request is malformed or meaningless   ValueError
//   EmptySelection     a sector selection matches no block       ValueError (InvalidRequest)
//   Unsupported        valid, but no lane implements it          NotImplementedError
//   DeviceUnavailable  device="gpu" with no usable device        RuntimeError
//   DeviceUnsupported  a block has no device lane                RuntimeError
//   ResourceLimit      the request does not fit memory / budget  MemoryError
//   ConvergenceError   an iterative solve did not converge       RuntimeError
// =============================================================================
#pragma once

#include <stdexcept>
#include <string>

namespace ed {

struct InvalidRequest : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

struct EmptySelection : InvalidRequest {
    using InvalidRequest::InvalidRequest;
};

struct Unsupported : std::logic_error {
    using std::logic_error::logic_error;
};

struct DeviceUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct DeviceUnsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ResourceLimit : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ConvergenceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ed
