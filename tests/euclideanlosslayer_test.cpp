#include <gtest/gtest.h>

#include "blob.hpp"
#include "euclideanlosslayer.hpp"

using namespace autoencoder;

TEST(EuclideanLossLayerTest, TestForwardCpu) {
  auto input = Blob<float>(10);
  auto target = Blob<float>(10);
  for (auto i = 0; i < target.width; ++i) {
    target.value(i) = 1.0f;
  }
  auto layer = EuclideanLossLayer();
  auto output = Blob<float>(10);
  auto out = Blobs<float>{&output};
  layer.ForwardCpu(Layer::Mode::kTrain, {&input, &target}, &out);

  for (auto i = 0; i < output.width; ++i) {
    EXPECT_FLOAT_EQ(0.5f, output.value(i));
    EXPECT_FLOAT_EQ(1.0f, output.difference(i));
  }
}

TEST(EuclideanLossLayerTest, TestBackwardCpu) {
  auto input = Blob<float>(10);
  auto target = Blob<float>(10);
  for (auto i = 0; i < target.width; ++i) {
    target.value(i) = 1.0f;
  }
  auto layer = EuclideanLossLayer();
  auto output = Blob<float>(10);
  auto in = Blobs<float>{&input, &target};
  auto out = Blobs<float>{&output};
  layer.ForwardCpu(Layer::Mode::kTrain, in, &out);
  layer.BackwardCpu(out, &in);

  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(1.0f, output.difference(i));
  }
}
