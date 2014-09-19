#include <gtest/gtest.h>
#include <random>

#include "blob.hpp"
#include "dropoutlayer.hpp"
#include "euclideanlosslayer.hpp"

using namespace autoencoder;

TEST(DropoutLayerTest, TestForwardCpu) {
  auto input = Blob<float>(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 1.0f;
  }
  auto generator = std::mt19937(123);
  auto layer = DropoutLayer<float>(0.5f, generator);
  auto output = Blob<float>(10);
  auto out = Blobs<float>{&output};
  layer.ForwardCpu(Mode::kTrain, {&input}, &out);

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
  auto input = Blob<float>(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 1.0f;
  }
  auto generator = std::mt19937(123);
  auto layer = DropoutLayer<float>(0.5f, generator);
  auto output = Blob<float>(10);
  auto in = Blobs<float>{&input};
  auto out = Blobs<float>{&output};
  layer.ForwardCpu(Mode::kTrain, in, &out);
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

TEST(DropoutLayerTest, TestGradient) {
  auto input = Blob<float>(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 1.0f;
  }
  auto generator = std::mt19937(123);
  auto layer = DropoutLayer<float>(0.5f, generator);
  auto loss_layer = EuclideanLossLayer<float>();
  auto in = Blobs<float>{&input};

  constexpr float kEpsilon = 1e-4;

  for (auto i = 0; i < input.width; ++i) {
    auto output = Blob<float>(10);
    auto out = Blobs<float>{&output};
    auto losses = Blob<float>(10);
    auto target = Blob<float>(10);
    for (auto j = 0; j < target.width; ++j) {
      target.value(j) = 1.0;
    }
    auto loss_in = Blobs<float>{&output, &target};
    auto loss_out = Blobs<float>{&losses};

    generator.seed(123);
    layer.ForwardCpu(Mode::kTrain, in, &out);
    loss_layer.ForwardCpu(Mode::kTrain, loss_in, &loss_out);
    loss_layer.BackwardCpu(loss_out, &loss_in);
    layer.Backward(out, &in);

    auto actual_partial_error_with_respect_to_input_i = input.difference(i);

    auto original_input_i = input.value(i);

    input.value(i) = original_input_i + kEpsilon;
    generator.seed(123);
    layer.ForwardCpu(Mode::kTrain, in, &out);
    auto loss_1 = loss_layer.ForwardCpu(Mode::kTrain, loss_in, &loss_out);

    input.value(i) = original_input_i - kEpsilon;
    generator.seed(123);
    layer.ForwardCpu(Mode::kTrain, in, &out);
    auto loss_0 = loss_layer.ForwardCpu(Mode::kTrain, loss_in, &loss_out);    

    input.value(i) = original_input_i;

    auto expected_partial_error_with_respect_to_input_i = (loss_1 - loss_0) / (2.0f * kEpsilon);

    EXPECT_NEAR(expected_partial_error_with_respect_to_input_i,
      actual_partial_error_with_respect_to_input_i, 1e-2);
  }
}
