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

TEST(ValuesTest, TestAddition) {
  auto vector_a = autoencoder::Values(10);
  for (auto i = 0; i < vector_a.width; ++i) {
    vector_a.value(i) = 1.0f;
  }
  auto vector_b = autoencoder::Values(10);
  for (auto i = 0; i < vector_b.width; ++i) {
    vector_b.value(i) = 2.0f;
  }
  auto vector_c = vector_a + vector_b;
  for (auto i = 0; i < vector_c.width; ++i) {
    EXPECT_FLOAT_EQ(3.0f, vector_c.value(i));
  }
}

TEST(ValuesTest, TestPrinting) {
  std::ostringstream out;
  out << autoencoder::Values(5);
  EXPECT_EQ("  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00", out.str());
}
