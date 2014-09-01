#include <cmath>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

#include "values.h"

TEST(ValuesTest, TestConstruction) {
  auto vector = autoencoder::Values(10);
  EXPECT_EQ(10, vector.width);
  for (auto i = 0; i < vector.width; ++i) {
    EXPECT_FLOAT_EQ(0.0f, vector.value(i));
  }
}

TEST(ValuesTest, TestElementAccess) {
  auto vector = autoencoder::Values(10);
  for (auto i = 0; i < vector.width; ++i) {
    vector.value(i) = i;
  }
  for (auto i = 0; i < vector.width; ++i) {
    EXPECT_FLOAT_EQ(i, vector.value(i));
  }
}

TEST(ValuesTest, TestElementAccessMultidimensional) {
  auto vector = autoencoder::Values(4, 4);
  for (auto i = 0; i < vector.width; ++i) {
    for (auto j = 0; j < vector.height; ++j) {
      vector.value(i, j) = (i < j ? -1.0f : 1.0f) * i * j;
    }
  }
  for (auto i = 0; i < vector.width; ++i) {
    for (auto j = 0; j < vector.height; ++j) {
      EXPECT_FLOAT_EQ((i < j ? -1.0f : 1.0f) * i * j, vector.value(i, j));
    }
  }
}

TEST(ValuesTest, TestPrinting) {
  std::ostringstream out;
  out << autoencoder::Values(5);
  EXPECT_EQ("  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00\n", out.str());
}
