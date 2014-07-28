#include <cmath>

#include "bloomfilter.h"

namespace voxels {

  template <int n, int pd>
  constexpr int BloomFilter<n, pd>::m() const {
    return -n * log(1.0 / pd) / log(2.0) / log(2.0);
  }

  template <int n, int pd>
  constexpr int BloomFilter<n, pd>::k() const {
    return static_cast<double>(m()) / n * log(2.0);
  }

}  // namespace voxels
