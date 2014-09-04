#include <gtest/gtest.h>

#include "blob.hpp"
#include "concatenatelayer.hpp"

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
  layer.ForwardCpu(in, &out);
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
