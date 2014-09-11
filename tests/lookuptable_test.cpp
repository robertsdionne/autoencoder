#include <gtest/gtest.h>
#include <random>

#include "blob.hpp"
#include "lookuptable.hpp"

TEST(LookupTableTest, TestForwardCpu) {
  auto words = std::vector<std::string>{"one", "two", "three", "four", "five"};
  auto vectors = std::vector<autoencoder::Blob>(5, autoencoder::Blob(10));
  for (auto i = 0; i < vectors.size(); ++i) {
    for (auto j = 0; j < vectors.at(0).width; ++j) {
      vectors.at(i).value(j) = i + 1;
    }
  }
  auto generator = std::mt19937(123);
  auto table = autoencoder::LookupTable(generator, words, vectors);
  auto input = std::vector<std::string>{"one", "three", "five"};
  auto output = autoencoder::Blobs{};
  table.ForwardCpu(input, &output);

  EXPECT_EQ(3, output.size());
  EXPECT_EQ(1, output.at(0)->value(0));
  EXPECT_EQ(3, output.at(1)->value(0));
  EXPECT_EQ(5, output.at(2)->value(0));
}