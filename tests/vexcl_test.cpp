#include <gtest/gtest.h>
#include <vector>
#include <vexcl/vexcl.hpp>

TEST(VexClTest, TestVexCl) {
  vex::Context context{vex::Filter::Type{CL_DEVICE_TYPE_GPU}};
  std::vector<float> h(10, 5.0f);
  vex::vector<float> d(context, 10);
  vex::copy(h, d);
  {
    auto mapped_pointer = d.map(0);
    for (auto i = 0; i < d.part_size(0); ++i) {
      EXPECT_FLOAT_EQ(5.0f, mapped_pointer[i]);
    }
  }
}
