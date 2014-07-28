#ifndef VOXELS_BLOOMFILTER_H_
#define VOXELS_BLOOMFILTER_H_

namespace voxels {

  template <int n, int pd>
  class BloomFilter {
  public:

    constexpr int m() const;

    constexpr int k() const;
  };

}  // namespace voxels

#endif  // VOXELS_BLOOMFILTER_H_
