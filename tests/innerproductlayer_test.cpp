#include <gtest/gtest.h>

#include "blob.hpp"
#include "euclideanlosslayer.hpp"
#include "innerproductlayer.hpp"

using namespace autoencoder;

TEST(InnerProductLayerTest, TestForwardCpu) {
  auto input = Blob<float>(4);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = i;
  }
  auto weights = Blob<float>(4, 3);
  for (auto i = 0; i < weights.height; ++i) {
    for (auto j = 0; j < weights.width; ++j) {
      weights.value(j, i) = (i + j);
    }
  }
  auto bias = Blob<float>(3);
  for (auto i = 0; i < bias.width; ++i) {
    bias.value(i) = i;
  }
  auto layer = InnerProductLayer<float>(weights, bias);
  auto output = Blob<float>(3);
  auto out = Blobs<float>{&output};
  layer.ForwardCpu(Mode::kTrain, {&input}, &out);

  EXPECT_FLOAT_EQ(14.0f, output.value(0));
  EXPECT_FLOAT_EQ(21.0f, output.value(1));
  EXPECT_FLOAT_EQ(28.0f, output.value(2));
}

TEST(InnerProductLayerTest, TestBackwardCpu) {
  auto input = Blob<float>(4);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = i;
  }
  auto weights = Blob<float>(4, 3);
  for (auto i = 0; i < weights.height; ++i) {
    for (auto j = 0; j < weights.width; ++j) {
      weights.value(j, i) = i + j;
    }
  }
  auto bias = Blob<float>(3);
  for (auto i = 0; i < bias.width; ++i) {
    bias.value(i) = i;
  }
  auto layer = InnerProductLayer<float>(weights, bias);
  auto output = Blob<float>(3);
  auto in = Blobs<float>{&input};
  auto out = Blobs<float>{&output};
  layer.ForwardCpu(Mode::kTrain, in, &out);
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

TEST(InnerProductLayerTest, TestGradient) {
  auto input = Blob<double>(4);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = i;
  }
  auto weights = Blob<double>(4, 3);
  for (auto i = 0; i < weights.height; ++i) {
    for (auto j = 0; j < weights.width; ++j) {
      weights.value(j, i) = (i + j) / 36.0f;
    }
  }
  auto bias = Blob<double>(3);
  for (auto i = 0; i < bias.width; ++i) {
    bias.value(i) = i / 6.0f;
  }
  auto in = Blobs<double>{&input};
  auto layer = InnerProductLayer<double>(weights, bias);
  auto loss_layer = EuclideanLossLayer<double>();

  constexpr double kEpsilon = 1e-4;
  constexpr double kTolerance = 1e-3;

  for (auto i = 0; i < input.width; ++i) {
    auto output = Blob<double>(3);
    auto out = Blobs<double>{&output};
    auto losses = Blob<double>(3);
    auto target = Blob<double>(3);
    for (auto j = 0; j < input.width; ++j) {
      input.difference(j) = 0.0;
    }
    auto loss_in = Blobs<double>{&output, &target};
    auto loss_out = Blobs<double>{&losses};

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
      actual_partial_error_with_respect_to_input_i, kTolerance);
  }

  for (auto i = 0; i < weights.height; ++i) {
    for (auto j = 0; j < weights.width; ++j) {
      auto output = Blob<double>(3);
      auto out = Blobs<double>{&output};
      auto losses = Blob<double>(3);
      auto target = Blob<double>(3);
      for (auto k = 0; k < weights.height; ++k) {
        for (auto l = 0; l < weights.width; ++l) {
          weights.difference(l, k) = 0.0;
        }
      }
      auto loss_in = Blobs<double>{&output, &target};
      auto loss_out = Blobs<double>{&losses};

      layer.ForwardCpu(Mode::kTrain, in, &out);
      loss_layer.Forward(Mode::kTrain, loss_in, &loss_out);
      loss_layer.Backward(loss_out, &loss_in);
      layer.BackwardCpu(out, &in);

      auto actual_partial_error_with_respect_to_weight_ij = weights.difference(j, i);

      auto original_weight_ij = weights.value(j, i);

      weights.value(j, i) = original_weight_ij + kEpsilon;
      layer.ForwardCpu(Mode::kTrain, in, &out);
      auto loss_1 = loss_layer.Forward(Mode::kTrain, loss_in, &loss_out);

      weights.value(j, i) = original_weight_ij - kEpsilon;
      layer.ForwardCpu(Mode::kTrain, in, &out);
      auto loss_0 = loss_layer.Forward(Mode::kTrain, loss_in, &loss_out);

      weights.value(j, i) = original_weight_ij;

      auto expected_partial_error_with_respect_to_weight_ij = (loss_1 - loss_0) / (2.0f * kEpsilon);

      EXPECT_NEAR(expected_partial_error_with_respect_to_weight_ij,
        actual_partial_error_with_respect_to_weight_ij, kTolerance);
    }
  }

  for (auto i = 0; i < bias.width; ++i) {
    auto output = Blob<double>(3);
    auto out = Blobs<double>{&output};
    auto losses = Blob<double>(3);
    auto target = Blob<double>(3);
    for (auto j = 0; j < bias.width; ++j) {
      bias.difference(j) = 0.0;
    }
    auto loss_in = Blobs<double>{&output, &target};
    auto loss_out = Blobs<double>{&losses};

    layer.ForwardCpu(Mode::kTrain, in, &out);
    loss_layer.Forward(Mode::kTrain, loss_in, &loss_out);
    loss_layer.Backward(loss_out, &loss_in);
    layer.BackwardCpu(out, &in);

    auto actual_partial_error_with_respect_to_bias_i = bias.difference(i);

    auto original_bias_i = bias.value(i);

    bias.value(i) = original_bias_i + kEpsilon;
    layer.ForwardCpu(Mode::kTrain, in, &out);
    auto loss_1 = loss_layer.Forward(Mode::kTrain, loss_in, &loss_out);

    bias.value(i) = original_bias_i - kEpsilon;
    layer.ForwardCpu(Mode::kTrain, in, &out);
    auto loss_0 = loss_layer.Forward(Mode::kTrain, loss_in, &loss_out);

    bias.value(i) = original_bias_i;

    auto expected_partial_error_with_respect_to_bias_i = (loss_1 - loss_0) / (2.0f * kEpsilon);

    EXPECT_NEAR(expected_partial_error_with_respect_to_bias_i,
      actual_partial_error_with_respect_to_bias_i, kTolerance);
  }
}
