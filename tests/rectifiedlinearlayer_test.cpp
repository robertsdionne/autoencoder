#include <gtest/gtest.h>

#include "rectifiedlinearlayer.h"
#include "vector.h"

TEST(RectifiedLinearLayerTest, TestForwardCpu) {
  auto input = autoencoder::Vector(10);
  for (auto i = 0; i < input.width; ++i) {
    input(i) = (i % 2) - (i % 2 == 0);
  }
  auto layer = autoencoder::RectifiedLinearLayer();
  auto output = autoencoder::Vector(10);
  layer.ForwardCpu(input, &output);
  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(i % 2, output(i));
  }
}
