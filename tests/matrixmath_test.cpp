#include <gtest/gtest.h>

#include "matrixmath.h"
#include "values.h"

TEST(MatrixMathTest, TestSgemm) {
  constexpr auto kM = 3, kK = 4, kN = 5;
  auto A = autoencoder::Values(kK, kM);
  for (auto i = 0; i < A.height; ++i) {
    for (auto j = 0; j < A.width; ++j) {
      A.value(j, i) = i + j;
    }
  }
  auto B = autoencoder::Values(kN, kK);
  for (auto i = 0; i < B.height; ++i) {
    for (auto j = 0; j < B.width; ++j) {
      B.value(j, i) = i + j;
    }
  }
  auto C = autoencoder::Values(kN, kM);
  autoencoder::Sgemm(1.0f, A, B, 0.0f, &C);

  EXPECT_FLOAT_EQ(14.0f, C.value(0, 0));
  EXPECT_FLOAT_EQ(20.0f, C.value(0, 1));
  EXPECT_FLOAT_EQ(26.0f, C.value(0, 2));

  EXPECT_FLOAT_EQ(20.0f, C.value(1, 0));
  EXPECT_FLOAT_EQ(30.0f, C.value(1, 1));
  EXPECT_FLOAT_EQ(40.0f, C.value(1, 2));

  EXPECT_FLOAT_EQ(26.0f, C.value(2, 0));
  EXPECT_FLOAT_EQ(40.0f, C.value(2, 1));
  EXPECT_FLOAT_EQ(54.0f, C.value(2, 2));

  EXPECT_FLOAT_EQ(32.0f, C.value(3, 0));
  EXPECT_FLOAT_EQ(50.0f, C.value(3, 1));
  EXPECT_FLOAT_EQ(68.0f, C.value(3, 2));

  EXPECT_FLOAT_EQ(38.0f, C.value(4, 0));
  EXPECT_FLOAT_EQ(60.0f, C.value(4, 1));
  EXPECT_FLOAT_EQ(82.0f, C.value(4, 2));
}
