#include <gtest/gtest.h>
#include <random>

#include "blob.hpp"
#include "partofspeechwordlayer.hpp"

template<typename Distribution, typename Generator>
void InitializeBlob(Distribution &distribution, Generator &generator, autoencoder::Blob *blob) {
  for (auto i = 0; i < blob->height; ++i) {
    for (auto j = 0; j < blob->width; ++j) {
      blob->value(j, i) = distribution(generator);
    }
  }
}

TEST(PartOfSpeechWordLayerTest, TestForwardCpu) {
  constexpr auto kRandomSeed = 123;
  std::mt19937 generator(kRandomSeed);
  std::uniform_real_distribution<float> uniform;
  std::uniform_real_distribution<float> uniform_symmetric(-1.0f, 1.0f);

  auto recurrent_input = autoencoder::Blob(10);
  auto word_input = autoencoder::Blob(10);
  auto classify_weights = autoencoder::Blob(10, 5);
  auto classify_bias = autoencoder::Blob(5);
  auto combine_weights = autoencoder::Blob(20, 10);
  auto combine_bias = autoencoder::Blob(10);

  InitializeBlob(uniform_symmetric, generator, &recurrent_input);
  InitializeBlob(uniform_symmetric, generator, &word_input);
  InitializeBlob(uniform, generator, &classify_weights);
  InitializeBlob(uniform, generator, &classify_bias);
  InitializeBlob(uniform, generator, &combine_weights);
  InitializeBlob(uniform, generator, &combine_bias);

  auto layer = autoencoder::PartOfSpeechWordLayer(
      0.5f, classify_weights, classify_bias, combine_weights, combine_bias, kRandomSeed);
  auto tag_output = autoencoder::Blob(5);
  auto recurrent_output = autoencoder::Blob(10);
  auto out = autoencoder::Blobs{&tag_output, &recurrent_output};
  layer.ForwardCpu({&recurrent_input, &word_input}, &out);

  auto sum = 0.0f;
  for (auto i = 0; i < tag_output.width; ++i) {
    sum += tag_output.value(i);
  }
  EXPECT_FLOAT_EQ(1.0f, sum);

  // Characterization tests.
  EXPECT_FLOAT_EQ(0.10213854f, tag_output.value(0));
  EXPECT_FLOAT_EQ(0.30351478f, tag_output.value(1));
  EXPECT_FLOAT_EQ(0.094782762f, tag_output.value(2));
  EXPECT_FLOAT_EQ(0.26085988f, tag_output.value(3));
  EXPECT_FLOAT_EQ(0.23870403f, tag_output.value(4));

  EXPECT_FLOAT_EQ(2.2393684f, recurrent_output.value(0));
  EXPECT_FLOAT_EQ(1.3972942f, recurrent_output.value(1));
  EXPECT_FLOAT_EQ(2.4804213f, recurrent_output.value(2));
  EXPECT_FLOAT_EQ(2.5403609f, recurrent_output.value(3));
  EXPECT_FLOAT_EQ(1.6428924f, recurrent_output.value(4));
  EXPECT_FLOAT_EQ(0.38085708f, recurrent_output.value(5));
  EXPECT_FLOAT_EQ(0.16046394f, recurrent_output.value(6));
  EXPECT_FLOAT_EQ(1.1690767f, recurrent_output.value(7));
  EXPECT_FLOAT_EQ(1.127956f, recurrent_output.value(8));
  EXPECT_FLOAT_EQ(2.1677952f, recurrent_output.value(9));
}

TEST(PartOfSpeechWordLayerTest, TestBackwardCpu) {
}
