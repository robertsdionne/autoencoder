#include <gtest/gtest.h>
#include <sstream>
#include <string>

#include "vector.h"

TEST(VectorTest, TestConstruction) {
  auto vector = autoencoder::Vector(10);
  EXPECT_EQ(10, vector.width);
  for (auto i = 0; i < vector.width; ++i) {
    EXPECT_FLOAT_EQ(0.0f, vector(i));
  }
}

TEST(VectorTest, TestElementAccess) {
  auto vector = autoencoder::Vector(10);
  for (auto i = 0; i < vector.width; ++i) {
    vector(i) = i;
  }
  for (auto i = 0; i < vector.width; ++i) {
    EXPECT_FLOAT_EQ(i, vector(i));
  }
}

TEST(VectorTest, TestAddition) {
  auto vector_a = autoencoder::Vector(10);
  for (auto i = 0; i < vector_a.width; ++i) {
    vector_a(i) = 1.0f;
  }
  auto vector_b = autoencoder::Vector(10);
  for (auto i = 0; i < vector_b.width; ++i) {
    vector_b(i) = 2.0f;
  }
  auto vector_c = vector_a + vector_b;
  for (auto i = 0; i < vector_c.width; ++i) {
    EXPECT_FLOAT_EQ(3.0f, vector_c(i));
  }
}

TEST(VectorTest, TestPrinting) {
  std::ostringstream out;
  out << autoencoder::Vector(5);
  EXPECT_EQ("  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00", out.str());
}
