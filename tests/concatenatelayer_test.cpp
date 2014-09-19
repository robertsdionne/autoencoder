#include <gtest/gtest.h>

#include "blob.hpp"
#include "concatenatelayer.hpp"
#include "euclideanlosslayer.hpp"

TEST(ConcatenateLayerTest, TestForwardCpu) {
  auto input1 = autoencoder::Blob(1);
  auto input2 = autoencoder::Blob(2);
  auto input3 = autoencoder::Blob(3);
  auto input4 = autoencoder::Blob(4);
  auto in = autoencoder::Blobs{&input1, &input2, &input3, &input4};
  for (auto i = 0; i < in.size(); ++i) {
    for (auto j = 0; j < in.at(i)->width; ++j) {
      in.at(i)->value(j) = i + 1;
    }
  }
  auto layer = autoencoder::ConcatenateLayer();
  auto output = autoencoder::Blob(10);
  auto out = autoencoder::Blobs{&output};
  layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, in, &out);

  EXPECT_FLOAT_EQ(1.0f, output.value(0));
  EXPECT_FLOAT_EQ(2.0f, output.value(1));
  EXPECT_FLOAT_EQ(2.0f, output.value(2));
  EXPECT_FLOAT_EQ(3.0f, output.value(3));
  EXPECT_FLOAT_EQ(3.0f, output.value(4));
  EXPECT_FLOAT_EQ(3.0f, output.value(5));
  EXPECT_FLOAT_EQ(4.0f, output.value(6));
  EXPECT_FLOAT_EQ(4.0f, output.value(7));
  EXPECT_FLOAT_EQ(4.0f, output.value(8));
  EXPECT_FLOAT_EQ(4.0f, output.value(9));
}

TEST(ConcatenateLayerTest, TestBackwardCpu) {
  auto output = autoencoder::Blob(10);
  output.difference(0) = 1.0f;
  output.difference(1) = 2.0f;
  output.difference(2) = 2.0f;
  output.difference(3) = 3.0f;
  output.difference(4) = 3.0f;
  output.difference(5) = 3.0f;
  output.difference(6) = 4.0f;
  output.difference(7) = 4.0f;
  output.difference(8) = 4.0f;
  output.difference(9) = 4.0f;
  auto layer = autoencoder::ConcatenateLayer();
  auto input1 = autoencoder::Blob(1);
  auto input2 = autoencoder::Blob(2);
  auto input3 = autoencoder::Blob(3);
  auto input4 = autoencoder::Blob(4);
  auto in = autoencoder::Blobs{&input1, &input2, &input3, &input4};
  layer.BackwardCpu({&output}, &in);

  for (auto i = 0; i < input1.width; ++i) {
    EXPECT_FLOAT_EQ(1.0f, input1.difference(i));
  }
  for (auto i = 0; i < input2.width; ++i) {
    EXPECT_FLOAT_EQ(2.0f, input2.difference(i));
  }
  for (auto i = 0; i < input3.width; ++i) {
    EXPECT_FLOAT_EQ(3.0f, input3.difference(i));
  }
  for (auto i = 0; i < input4.width; ++i) {
    EXPECT_FLOAT_EQ(4.0f, input4.difference(i));
  }
}

TEST(ConcatenateLayerTest, TestGradient) {
  auto input1 = autoencoder::Blob(1);
  auto input2 = autoencoder::Blob(2);
  auto input3 = autoencoder::Blob(3);
  auto input4 = autoencoder::Blob(4);
  auto in = autoencoder::Blobs{&input1, &input2, &input3, &input4};
  for (auto i = 0; i < in.size(); ++i) {
    for (auto j = 0; j < in.at(i)->width; ++j) {
      in.at(i)->value(j) = i + 1;
    }
  }

  auto layer = autoencoder::ConcatenateLayer();
  auto loss_layer = autoencoder::EuclideanLossLayer();

  constexpr auto kEpsilon = 1e-4;

  for (auto input : in) {
    for (auto i = 0; i < input->width; ++i) {
      auto output = autoencoder::Blob(10);
      auto out = autoencoder::Blobs{&output};
      auto losses = autoencoder::Blob(10);
      auto target = autoencoder::Blob(10);
      for (auto j = 0; j < target.width; ++j) {
        target.value(j) = 1.0f;
      }
      auto loss_in = autoencoder::Blobs{&output, &target};
      auto loss_out = autoencoder::Blobs{&losses};

      layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, in, &out);
      loss_layer.Forward(autoencoder::Layer::Mode::kTrain, loss_in, &loss_out);
      loss_layer.Backward(loss_out, &loss_in);
      layer.BackwardCpu(out, &in);

      auto actual_partial_error_with_respect_to_input_i = input->difference(i);

      auto original_input_i = input->value(i);

      input->value(i) = original_input_i + kEpsilon;
      layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, in, &out);
      auto loss_1 = loss_layer.Forward(autoencoder::Layer::Mode::kTrain, loss_in, &loss_out);

      input->value(i) = original_input_i - kEpsilon;
      layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, in, &out);
      auto loss_0 = loss_layer.Forward(autoencoder::Layer::Mode::kTrain, loss_in, &loss_out);

      input->value(i) = original_input_i;

      auto expected_partial_error_with_respect_to_input_i = (loss_1 - loss_0) / (2.0f * kEpsilon);

      EXPECT_NEAR(expected_partial_error_with_respect_to_input_i,
        actual_partial_error_with_respect_to_input_i, 1e-2);
    }
  }
}
