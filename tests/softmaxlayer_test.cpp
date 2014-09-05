#include <cmath>
#include <gtest/gtest.h>

#include "blob.hpp"
#include "softmaxlayer.hpp"

TEST(SoftmaxLayerTest, TestForwardCpu) {
  auto input = autoencoder::Blob(8);
  for (auto i = 0; i < input.width; ++i) {
    input.value(i) = i;
  }
  auto layer = autoencoder::SoftmaxLayer();
  auto output = autoencoder::Blob(8);
  auto out = autoencoder::Blobs{&output};
  layer.ForwardCpu({&input}, &out);

  EXPECT_FLOAT_EQ(0.00057661277f, output.value(0));
  EXPECT_FLOAT_EQ(0.0015673961f, output.value(1));
  EXPECT_FLOAT_EQ(0.0042606243f, output.value(2));
  EXPECT_FLOAT_EQ(0.011581577f, output.value(3));
  EXPECT_FLOAT_EQ(0.031481992f, output.value(4));
  EXPECT_FLOAT_EQ(0.085576929f, output.value(5));
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
  layer.ForwardCpu(in, &out);
  for (auto i = 0; i < output.width; ++i) {
    output.difference(i) = (i == 2);
  }
  layer.BackwardCpu(out, &in);

  EXPECT_FLOAT_EQ(-2.4567305e-06f, input.difference(0));
  EXPECT_FLOAT_EQ(-6.6780858e-06f, input.difference(1));
  EXPECT_FLOAT_EQ(0.0042424714f, input.difference(2));
  EXPECT_FLOAT_EQ(-4.9344751e-05f, input.difference(3));
  EXPECT_FLOAT_EQ(-0.00013413295f, input.difference(4));
  EXPECT_FLOAT_EQ(-0.00036461113f, input.difference(5));
  EXPECT_FLOAT_EQ(-0.00099111581f, input.difference(6));
  EXPECT_FLOAT_EQ(-0.0026941323f, input.difference(7));
}
