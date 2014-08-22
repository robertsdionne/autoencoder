#include <gtest/gtest.h>
#include <sstream>
#include <string>

#include "matrix.h"

TEST(MatrixTest, TestConstruction) {
  auto matrix = autoencoder::Matrix(5, 10);
  EXPECT_EQ(5, matrix.height);
  EXPECT_EQ(10, matrix.width);
  for (auto i = 0; i < matrix.height; ++i) {
    for (auto j = 0; j < matrix.width; ++j) {
      EXPECT_FLOAT_EQ(0.0f, matrix(i, j));
    }
  }
}

TEST(MatrixTest, TestElementAccess) {
  auto matrix = autoencoder::Matrix(10, 10);
  for (auto i = 0; i < matrix.height; ++i) {
    for (auto j = 0; j < matrix.width; ++j) {
      matrix(i, j) = 10.0 * i + j;
    }
  }
  for (auto i = 0; i < matrix.height; ++i) {
    for (auto j = 0; j < matrix.width; ++j) {
      EXPECT_FLOAT_EQ(10.0f * i + j, matrix(i, j));
    }
  }
}

TEST(MatrixTest, TestMatrixMatrixAddition) {
  auto matrix_a = autoencoder::Matrix(2, 2);
  for (auto i = 0; i < matrix_a.height; ++i) {
    for (auto j = 0; j < matrix_a.width; ++j) {
      matrix_a(i, j) = 1.0f;
    }
  }
  auto matrix_b = autoencoder::Matrix(2, 2);
  for (auto i = 0; i < matrix_b.height; ++i) {
    for (auto j = 0; j < matrix_b.width; ++j) {
      matrix_b(i, j) = 2.0f;
    }
  }
  auto matrix_c = matrix_a + matrix_b;
  for (auto i = 0; i < matrix_c.height; ++i) {
    for (auto j = 0; j < matrix_c.width; ++j) {
      EXPECT_FLOAT_EQ(3.0f, matrix_c(i, j));
    }
  }
}

TEST(MatrixTest, TestMatrixVectorMultiplication) {
  auto matrix = autoencoder::Matrix(3, 3);
  for (auto i = 0; i < matrix.height; ++i) {
    matrix(i, i) = 1.0f;
  }
  auto vector = autoencoder::Vector(3);
  for (auto i = 0; i < vector.width; ++i) {
    vector(i) = i;
  }
  auto result = matrix * vector;
  for (auto i = 0; i < result.width; ++i) {
    EXPECT_FLOAT_EQ(vector(i), result(i));
  }
}

TEST(MatrixTest, TestPrinting) {
  std::ostringstream out;
  out << autoencoder::Matrix(3, 3);
  EXPECT_EQ(R"(  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00
)", out.str());
}
