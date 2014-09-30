#include <gtest/gtest.h>

#include "blob.hpp"
#include "cpudevice.hpp"
#include "euclideanlosslayer.hpp"
#include "vexcldevice.hpp"

using namespace autoencoder;

TEST(EuclideanLossLayerTest, TestForwardGpu) {
  auto input = Blob<float>(10);
  auto target = Blob<float>(10);
  for (auto i = 0; i < target.width; ++i) {
    target.value(i) = 1.0f;
  }
  auto device = VexClDevice<float>();
  auto layer = EuclideanLossLayer<float>(device);
  auto output = Blob<float>(10);
  auto out = Blobs<float>{&output};
  device.Initialize(input);
  device.Initialize(target);
  device.Initialize(output);
  device.Ship(input);
  device.Ship(target);
  device.Ship(output);
  layer.ForwardXpu(Mode::kTrain, {&input, &target}, &out);
  device.Retrieve(output);

  for (auto i = 0; i < output.width; ++i) {
    EXPECT_FLOAT_EQ(0.5f, output.value(i));
    EXPECT_FLOAT_EQ(1.0f, output.difference(i));
  }
}

TEST(EuclideanLossLayerTest, TestBackwardGpu) {
  auto input = Blob<float>(10);
  auto target = Blob<float>(10);
  for (auto i = 0; i < target.width; ++i) {
    target.value(i) = 1.0f;
  }
  auto device = VexClDevice<float>();
  auto layer = EuclideanLossLayer<float>(device);
  auto output = Blob<float>(10);
  auto in = Blobs<float>{&input, &target};
  auto out = Blobs<float>{&output};
  device.Initialize(input);
  device.Initialize(target);
  device.Initialize(output);
  device.Ship(input);
  device.Ship(target);
  device.Ship(output);
  layer.ForwardXpu(Mode::kTrain, in, &out);
  layer.BackwardXpu(out, &in);
  device.Retrieve(output);

  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(1.0f, output.difference(i));
  }
}

TEST(EuclideanLossLayerTest, TestForwardCpu) {
  auto input = Blob<float>(10);
  auto target = Blob<float>(10);
  for (auto i = 0; i < target.width; ++i) {
    target.value(i) = 1.0f;
  }
  auto device = CpuDevice<float>();
  auto layer = EuclideanLossLayer<float>(device);
  auto output = Blob<float>(10);
  auto out = Blobs<float>{&output};
  layer.ForwardXpu(Mode::kTrain, {&input, &target}, &out);

  for (auto i = 0; i < output.width; ++i) {
    EXPECT_FLOAT_EQ(0.5f, output.value(i));
    EXPECT_FLOAT_EQ(1.0f, output.difference(i));
  }
}

TEST(EuclideanLossLayerTest, TestBackwardCpu) {
  auto input = Blob<float>(10);
  auto target = Blob<float>(10);
  for (auto i = 0; i < target.width; ++i) {
    target.value(i) = 1.0f;
  }
  auto device = CpuDevice<float>();
  auto layer = EuclideanLossLayer<float>(device);
  auto output = Blob<float>(10);
  auto in = Blobs<float>{&input, &target};
  auto out = Blobs<float>{&output};
  layer.ForwardXpu(Mode::kTrain, in, &out);
  layer.BackwardXpu(out, &in);

  for (auto i = 0; i < input.width; ++i) {
    EXPECT_FLOAT_EQ(1.0f, output.difference(i));
  }
}
