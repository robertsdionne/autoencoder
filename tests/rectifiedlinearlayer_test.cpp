#include <gtest/gtest.h>
#include <vector>

#include "blob.hpp"
#include "cpudevice.hpp"
#include "euclideanlosslayer.hpp"
#include "rectifiedlinearlayer.hpp"

using namespace autoencoder;

TEST(RectifiedLinearLayer, TestForwardGpu) {
  auto input = Blob<float>(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto layer = RectifiedLinearLayer<float>();
  auto output = Blob<float>(10);
  auto out = Blobs<float>{&output};
  layer.ForwardGpu(Mode::kTrain, {&input}, &out);

  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(2.0f * (i % 2), output.value(i));
  }
}

TEST(RectifiedLinearLayerTest, TestForwardCpu) {
  auto input = Blob<float>(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto layer = RectifiedLinearLayer<float>();
  auto output = Blob<float>(10);
  auto out = Blobs<float>{&output};
  layer.ForwardCpu(Mode::kTrain, {&input}, &out);

  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(2.0f * (i % 2), output.value(i));
  }
}

TEST(RectifiedLinearLayerTest, TestBackwardCpu) {
  auto output = Blob<float>(10);
  for (auto i = 0; i < output.width; ++i) {
    output.difference(i) = 1.0f;
  }
  auto layer = RectifiedLinearLayer<float>();
  auto input = Blob<float>(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto in = Blobs<float>{&input};
  layer.BackwardCpu({&output}, &in);

  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(i % 2, input.difference(i));
  }
}

TEST(RectifiedLinearLayerTest, TestGradient) {
  auto input = Blob<float>(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto in = Blobs<float>{&input};
  auto layer = RectifiedLinearLayer<float>();
  auto device = CpuDevice<float>();
  auto loss_layer = EuclideanLossLayer<float>(device);

  constexpr float kEpsilon = 1e-4;

  for (auto i = 0; i < input.width; ++i) {
    auto output = Blob<float>(10);
    auto out = Blobs<float>{&output};
    auto losses = Blob<float>(10);
    auto target = Blob<float>(10);
    for (auto j = 0; j < input.width; ++j) {
      input.difference(j) = 0.0;
    }
    auto loss_in = Blobs<float>{&output, &target};
    auto loss_out = Blobs<float>{&losses};

    layer.ForwardCpu(Mode::kTrain, in, &out);
    loss_layer.Forward(Mode::kTrain, loss_in, &loss_out);
    loss_layer.Backward(loss_out, &loss_in);
    layer.BackwardCpu(out, &in);

    auto actual_partial_error_with_respect_to_input_i = input.difference(i);

    auto original_input_i = input.value(i);

    input.value(i) = original_input_i + kEpsilon;
    layer.ForwardCpu(Mode::kTrain, in, &out);
    auto loss_1 = loss_layer.Forward(Mode::kTrain, loss_in, &loss_out);

    input.value(i) = original_input_i - kEpsilon;
    layer.ForwardCpu(Mode::kTrain, in, &out);
    auto loss_0 = loss_layer.Forward(Mode::kTrain, loss_in, &loss_out);

    input.value(i) = original_input_i;

    auto expected_partial_error_with_respect_to_input_i = (loss_1 - loss_0) / (2.0f * kEpsilon);

    EXPECT_NEAR(expected_partial_error_with_respect_to_input_i,
      actual_partial_error_with_respect_to_input_i, 1e-2);
  }
}
