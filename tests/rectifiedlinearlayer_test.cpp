#include <gtest/gtest.h>
#include <vector>

#include "blob.hpp"
#include "euclideanlosslayer.hpp"
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

TEST(RectifiedLinearLayerTest, TestGradient) {
  auto input = Blob(10);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = 2.0f * (i % 2) - 2.0f * (i % 2 == 0);
  }
  auto in = Blobs{&input};
  auto layer = RectifiedLinearLayer();
  auto loss_layer = EuclideanLossLayer();

  constexpr float kEpsilon = 1e-4;

  for (auto i = 0; i < input.width; ++i) {
    auto output = Blob(10);
    auto out = Blobs{&output};
    auto losses = Blob(10);
    auto target = Blob(10);
    for (auto j = 0; j < input.width; ++j) {
      input.difference(j) = 0.0;
    }
    auto loss_in = Blobs{&output, &target};
    auto loss_out = Blobs{&losses};

    layer.ForwardCpu(Layer::Mode::kTrain, in, &out);
    loss_layer.Forward(Layer::Mode::kTrain, loss_in, &loss_out);
    loss_layer.Backward(loss_out, &loss_in);
    layer.BackwardCpu(out, &in);

    auto actual_partial_error_with_respect_to_input_i = input.difference(i);

    auto original_input_i = input.value(i);

    input.value(i) = original_input_i + kEpsilon;
    layer.ForwardCpu(Layer::Mode::kTrain, in, &out);
    auto loss_1 = loss_layer.Forward(Layer::Mode::kTrain, loss_in, &loss_out);

    input.value(i) = original_input_i - kEpsilon;
    layer.ForwardCpu(Layer::Mode::kTrain, in, &out);
    auto loss_0 = loss_layer.Forward(Layer::Mode::kTrain, loss_in, &loss_out);

    input.value(i) = original_input_i;

    auto expected_partial_error_with_respect_to_input_i = (loss_1 - loss_0) / (2.0f * kEpsilon);

    EXPECT_NEAR(expected_partial_error_with_respect_to_input_i,
      actual_partial_error_with_respect_to_input_i, 1e-2);
  }
}
