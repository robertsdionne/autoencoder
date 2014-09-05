#include <gtest/gtest.h>

#include "blob.hpp"
#include "innerproductlayer.hpp"

TEST(InnerProductLayerTest, TestForwardCpu) {
  auto input = autoencoder::Blob(4);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = i;
  }
  auto weights = autoencoder::Blob(4, 3);
  for (auto i = 0; i < weights.height; ++i) {
    for (auto j = 0; j < weights.width; ++j) {
      weights.value(j, i) = i + j;
    }
  }
  auto bias = autoencoder::Blob(3);
  for (auto i = 0; i < bias.width; ++i) {
    bias.value(i) = i;
  }
  auto layer = autoencoder::InnerProductLayer(weights, bias);
  auto output = autoencoder::Blob(3);
  auto out = autoencoder::Blobs{&output};
  layer.ForwardCpu({&input}, &out);

  EXPECT_FLOAT_EQ(14.0f, output.value(0));
  EXPECT_FLOAT_EQ(21.0f, output.value(1));
  EXPECT_FLOAT_EQ(28.0f, output.value(2));
}

TEST(InnerProductLayerTest, TestBackwardCpu) {
  auto input = autoencoder::Blob(4);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = i;
  }
  auto weights = autoencoder::Blob(4, 3);
  for (auto i = 0; i < weights.height; ++i) {
    for (auto j = 0; j < weights.width; ++j) {
      weights.value(j, i) = i + j;
    }
  }
  auto bias = autoencoder::Blob(3);
  for (auto i = 0; i < bias.width; ++i) {
    bias.value(i) = i;
  }
  auto layer = autoencoder::InnerProductLayer(weights, bias);
  auto output = autoencoder::Blob(3);
  auto in = autoencoder::Blobs{&input};
  auto out = autoencoder::Blobs{&output};
  layer.ForwardCpu(in, &out);
  for (auto i = 0; i < output.width; ++i) {
    output.difference(i) = 1.0f;
  }
  layer.BackwardCpu(out, &in);

  EXPECT_FLOAT_EQ(0.0f, weights.difference(0, 0));
  EXPECT_FLOAT_EQ(0.0f, weights.difference(0, 1));
  EXPECT_FLOAT_EQ(0.0f, weights.difference(0, 2));
  EXPECT_FLOAT_EQ(1.0f, weights.difference(1, 0));
  EXPECT_FLOAT_EQ(1.0f, weights.difference(1, 1));
  EXPECT_FLOAT_EQ(1.0f, weights.difference(1, 2));
  EXPECT_FLOAT_EQ(2.0f, weights.difference(2, 0));
  EXPECT_FLOAT_EQ(2.0f, weights.difference(2, 1));
  EXPECT_FLOAT_EQ(2.0f, weights.difference(2, 2));
  EXPECT_FLOAT_EQ(3.0f, weights.difference(3, 0));
  EXPECT_FLOAT_EQ(3.0f, weights.difference(3, 1));
  EXPECT_FLOAT_EQ(3.0f, weights.difference(3, 2));

  EXPECT_FLOAT_EQ(1.0f, bias.difference(0));
  EXPECT_FLOAT_EQ(1.0f, bias.difference(1));
  EXPECT_FLOAT_EQ(1.0f, bias.difference(2));

  EXPECT_FLOAT_EQ(3.0f, input.difference(0));
  EXPECT_FLOAT_EQ(6.0f, input.difference(1));
  EXPECT_FLOAT_EQ(9.0f, input.difference(2));
  EXPECT_FLOAT_EQ(12.0f, input.difference(3));
}
