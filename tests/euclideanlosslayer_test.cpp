#include <gtest/gtest.h>

#include "blob.hpp"
#include "euclideanlosslayer.hpp"

TEST(EuclideanLossLayerTest, TestForwardCpu) {
  auto input = autoencoder::Blob(10);
  auto target = autoencoder::Blob(10);
  for (auto i = 0; i < target.width; ++i) {
    target.value(i) = 1.0f;
  }
  auto layer = autoencoder::EuclideanLossLayer();
  auto output = autoencoder::Blob(10);
  auto out = autoencoder::Blobs{&output};
  layer.ForwardCpu({&input, &target}, &out);
  for (auto i = 0; i < output.width; ++i) {
    EXPECT_FLOAT_EQ(0.5f, output.value(i));
    EXPECT_FLOAT_EQ(1.0f, output.difference(i));
  }
}

TEST(EuclideanLossLayerTest, TestBackwardCpu) {
  auto input = autoencoder::Blob(10);
  auto target = autoencoder::Blob(10);
  for (auto i = 0; i < target.width; ++i) {
    target.value(i) = 1.0f;
  }
  auto layer = autoencoder::EuclideanLossLayer();
  auto output = autoencoder::Blob(10);
  auto in = autoencoder::Blobs{&input, &target};
  auto out = autoencoder::Blobs{&output};
  layer.ForwardCpu(in, &out);
  layer.BackwardCpu(out, &in);
  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(1.0f, output.difference(i));
  }
}
