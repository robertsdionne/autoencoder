#ifndef VOXELS_BLOOMFILTER_H_
#define VOXELS_BLOOMFILTER_H_

#include <cmath>

namespace voxels {

  template <int n, int pd>
  class BloomFilter {
  public:

    constexpr int m() const {
      return -n * log(1.0 / pd) / log(2.0) / log(2.0);
    }

    constexpr int k() const {
      return static_cast<double>(m()) / n * log(2.0);
    }
  };

}  // namespace voxels

#endif  // VOXELS_BLOOMFILTER_H_
