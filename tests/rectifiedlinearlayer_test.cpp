#include <gtest/gtest.h>

#include "rectifiedlinearlayer.h"
#include "vector.h"

TEST(RectifiedLinearLayerTest, TestForwardCpu) {
  auto input = autoencoder::Vector(10);
  for (auto i = 0; i < input.width; ++i) {
    input(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto layer = autoencoder::RectifiedLinearLayer();
  auto output = autoencoder::Vector(10);
  layer.ForwardCpu(input, &output);
  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(2.0f * (i % 2), output(i));
  }
}

TEST(RectifiedLinearLayerTest, TestBackwardCpu) {
  auto output = autoencoder::Vector(10);
  for (auto i = 0; i < output.width; ++i) {
    output(i) = 2.0f * (i % 2);
  }
  auto layer = autoencoder::RectifiedLinearLayer();
  auto input = autoencoder::Vector(10);
  layer.BackwardCpu(output, &input);
  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(i % 2, input(i));
  }
}
