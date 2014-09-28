#include <gtest/gtest.h>
#include <valarray>
#include <vector>
#include <vexcl/vexcl.hpp>

TEST(VexClTest, TestVexCl) {
  vex::Context context{vex::Filter::Type{CL_DEVICE_TYPE_GPU}};
  std::valarray<float> h(5.0f, 10);
  vex::vector<float> d(context, 10);
  vex::copy(std::begin(h), std::end(h), d.begin());
  {
    auto mapped_pointer = d.map(0);
    for (auto i = 0; i < d.part_size(0); ++i) {
      EXPECT_FLOAT_EQ(5.0f, mapped_pointer[i]);
    }
  }
}
