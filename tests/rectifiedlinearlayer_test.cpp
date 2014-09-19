#include <gtest/gtest.h>
#include <vector>

#include "blob.hpp"
#include "rectifiedlinearlayer.hpp"

using namespace autoencoder;

TEST(RectifiedLinearLayerTest, TestForwardCpu) {
  auto input = Blob(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto layer = RectifiedLinearLayer();
  auto output = Blob(10);
  auto out = Blobs{&output};
  layer.ForwardCpu(Layer::Mode::kTrain, {&input}, &out);

  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(2.0f * (i % 2), output.value(i));
  }
}

TEST(RectifiedLinearLayerTest, TestBackwardCpu) {
  auto output = Blob(10);
  for (auto i = 0; i < output.width; ++i) {
    output.difference(i) = 1.0f;
  }
  auto layer = RectifiedLinearLayer();
  auto input = Blob(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto in = Blobs{&input};
  layer.BackwardCpu({&output}, &in);

  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(i % 2, input.difference(i));
  }
}
