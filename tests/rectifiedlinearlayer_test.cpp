#include <gtest/gtest.h>
#include <vector>

#include "blob.hpp"
#include "rectifiedlinearlayer.hpp"

TEST(RectifiedLinearLayerTest, TestForwardCpu) {
  auto input = autoencoder::Blob(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto layer = autoencoder::RectifiedLinearLayer();
  auto output = autoencoder::Blob(10);
  auto out = std::vector<autoencoder::Blob *>{&output};
  layer.ForwardCpu({&input}, &out);
  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(2.0f * (i % 2), output.value(i));
  }
}

TEST(RectifiedLinearLayerTest, TestBackwardCpu) {
  auto output = autoencoder::Blob(10);
  for (auto i = 0; i < output.width; ++i) {
    output.difference(i) = 1.0f;
  }
  auto layer = autoencoder::RectifiedLinearLayer();
  auto input = autoencoder::Blob(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto in = std::vector<autoencoder::Blob *>{&input};
  layer.BackwardCpu({&output}, &in);
  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(i % 2, input.difference(i));
  }
}
