#include <gtest/gtest.h>
#include <random>

#include "blob.hpp"
#include "dropoutlayer.hpp"

TEST(DropoutLayerTest, TestForwardCpu) {
  auto input = autoencoder::Blob(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 1.0f;
  }
  auto generator = std::mt19937(123);
  auto layer = autoencoder::DropoutLayer(0.5f, generator);
  auto output = autoencoder::Blob(10);
  auto out = autoencoder::Blobs{&output};
  layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, {&input}, &out);

  // TODO(robertsdionne): remove dependency upon random number generator code with a mock.
  EXPECT_FLOAT_EQ(0.0f, output.value(0));
  EXPECT_FLOAT_EQ(2.0f, output.value(1));
  EXPECT_FLOAT_EQ(0.0f, output.value(2));
  EXPECT_FLOAT_EQ(0.0f, output.value(3));
  EXPECT_FLOAT_EQ(2.0f, output.value(4));
  EXPECT_FLOAT_EQ(0.0f, output.value(5));
  EXPECT_FLOAT_EQ(2.0f, output.value(6));
  EXPECT_FLOAT_EQ(0.0f, output.value(7));
  EXPECT_FLOAT_EQ(2.0f, output.value(8));
  EXPECT_FLOAT_EQ(2.0f, output.value(9));
}

TEST(DropoutLayerTest, TestBackwardCpu) {
  auto input = autoencoder::Blob(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 1.0f;
  }
  auto generator = std::mt19937(123);
  auto layer = autoencoder::DropoutLayer(0.5f, generator);
  auto output = autoencoder::Blob(10);
  auto in = autoencoder::Blobs{&input};
  auto out = autoencoder::Blobs{&output};
  layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, in, &out);
  for (auto i = 0; i < output.width; ++i) {
    output.difference(i) = 1.0f;
  }
  layer.BackwardCpu(out, &in);

  // TODO(robertsdionne): remove dependency upon random number generator code with a mock.
  EXPECT_FLOAT_EQ(0.0f, input.difference(0));
  EXPECT_FLOAT_EQ(2.0f, input.difference(1));
  EXPECT_FLOAT_EQ(0.0f, input.difference(2));
  EXPECT_FLOAT_EQ(0.0f, input.difference(3));
  EXPECT_FLOAT_EQ(2.0f, input.difference(4));
  EXPECT_FLOAT_EQ(0.0f, input.difference(5));
  EXPECT_FLOAT_EQ(2.0f, input.difference(6));
  EXPECT_FLOAT_EQ(0.0f, input.difference(7));
  EXPECT_FLOAT_EQ(2.0f, input.difference(8));
  EXPECT_FLOAT_EQ(2.0f, input.difference(9));
}
