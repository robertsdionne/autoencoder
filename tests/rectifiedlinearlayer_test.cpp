#include <gtest/gtest.h>

#include "rectifiedlinearlayer.h"
#include "values.h"

TEST(RectifiedLinearLayerTest, TestForwardCpu) {
  auto input = autoencoder::Values(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto layer = autoencoder::RectifiedLinearLayer();
  auto output = autoencoder::Values(10);
  layer.ForwardCpu(input, &output);
  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(2.0f * (i % 2), output.value(i));
  }
}

TEST(RectifiedLinearLayerTest, TestBackwardCpu) {
  auto output = autoencoder::Values(10);
  for (auto i = 0; i < output.width; ++i) {
    output.difference(i) = 2.0f * (i % 2);
  }
  auto layer = autoencoder::RectifiedLinearLayer();
  auto input = autoencoder::Values(10);
  layer.BackwardCpu(output, &input);
  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(i % 2, input.difference(i));
  }
}
