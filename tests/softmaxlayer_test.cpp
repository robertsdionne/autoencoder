#include <cmath>
#include <gtest/gtest.h>

#include "blob.hpp"
#include "euclideanlosslayer.hpp"
#include "softmaxlayer.hpp"

TEST(SoftmaxLayerTest, TestForwardCpu) {
  auto input = autoencoder::Blob(8);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = i;
  }
  auto layer = autoencoder::SoftmaxLayer();
  auto output = autoencoder::Blob(8);
  auto out = autoencoder::Blobs{&output};
  layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, {&input}, &out);

  EXPECT_FLOAT_EQ(0.00057668809f, output.value(0));
  EXPECT_FLOAT_EQ(0.0015674712f, output.value(1));
  EXPECT_FLOAT_EQ(0.0042606993f, output.value(2));
  EXPECT_FLOAT_EQ(0.011581651f, output.value(3));
  EXPECT_FLOAT_EQ(0.031482063f, output.value(4));
  EXPECT_FLOAT_EQ(0.085576989f, output.value(5));
  EXPECT_FLOAT_EQ(0.23262221f, output.value(6));
  EXPECT_FLOAT_EQ(0.63233274f, output.value(7));

  auto sum = 0.0f;
  for (auto i = 0; i < output.width; ++i) {
    sum += output.value(i);
  }
  EXPECT_FLOAT_EQ(1.0f, sum);
}

TEST(SoftmaxLayerTest, TestBackwardCpu) {
  auto input = autoencoder::Blob(8);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = i;
  }
  auto layer = autoencoder::SoftmaxLayer();
  auto output = autoencoder::Blob(8);
  auto in = autoencoder::Blobs{&input};
  auto out = autoencoder::Blobs{&output};
  layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, in, &out);
  for (auto i = 0; i < output.width; ++i) {
    output.difference(i) = (i == 2);
  }
  layer.BackwardCpu(out, &in);

  EXPECT_FLOAT_EQ(-2.4570945e-06f, input.difference(0));
  EXPECT_FLOAT_EQ(-6.6785233e-06f, input.difference(1));
  EXPECT_FLOAT_EQ(0.0042425455f, input.difference(2));
  EXPECT_FLOAT_EQ(-4.934593e-05f, input.difference(3));
  EXPECT_FLOAT_EQ(-0.0001341356f, input.difference(4));
  EXPECT_FLOAT_EQ(-0.00036461782f, input.difference(5));
  EXPECT_FLOAT_EQ(-0.00099113351f, input.difference(6));
  EXPECT_FLOAT_EQ(-0.0026941793f, input.difference(7));
}

TEST(SoftmaxLayerTest, TestGradient) {
  auto input = autoencoder::Blob(8);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = i;
  }
  auto in = autoencoder::Blobs{&input};
  auto layer = autoencoder::SoftmaxLayer();
  auto loss_layer = autoencoder::EuclideanLossLayer();

  constexpr float kEpsilon = 1e-4;

  for (auto i = 0; i < input.width; ++i) {
    auto output = autoencoder::Blob(8);
    auto out = autoencoder::Blobs{&output};
    auto losses = autoencoder::Blob(8);
    auto target = autoencoder::Blob(8);
    for (auto j = 0; j < input.width; ++j) {
      input.difference(j) = 0.0;
    }
    auto loss_in = autoencoder::Blobs{&output, &target};
    auto loss_out = autoencoder::Blobs{&losses};

    layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, in, &out);
    loss_layer.Forward(autoencoder::Layer::Mode::kTrain, loss_in, &loss_out);
    loss_layer.Backward(loss_out, &loss_in);
    layer.BackwardCpu(out, &in);

    auto actual_partial_error_with_respect_to_input_i = input.difference(i);

    auto original_input_i = input.value(i);

    input.value(i) = original_input_i + kEpsilon;
    layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, in, &out);
    auto loss_1 = loss_layer.Forward(autoencoder::Layer::Mode::kTrain, loss_in, &loss_out);

    input.value(i) = original_input_i - kEpsilon;
    layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, in, &out);
    auto loss_0 = loss_layer.Forward(autoencoder::Layer::Mode::kTrain, loss_in, &loss_out);

    input.value(i) = original_input_i;

    auto expected_partial_error_with_respect_to_input_i = (loss_1 - loss_0) / (2.0f * kEpsilon);

    EXPECT_NEAR(expected_partial_error_with_respect_to_input_i,
      actual_partial_error_with_respect_to_input_i, 1e-2);
  }
}
