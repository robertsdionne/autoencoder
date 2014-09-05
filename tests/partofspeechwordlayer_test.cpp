#include <gtest/gtest.h>
#include <random>

#include "blob.hpp"
#include "partofspeechwordlayer.hpp"

constexpr auto kRandomSeed = 123;

template<typename Distribution, typename Generator>
void InitializeBlob(Distribution &distribution, Generator &generator, autoencoder::Blob *blob) {
  for (auto i = 0; i < blob->height; ++i) {
    for (auto j = 0; j < blob->width; ++j) {
      blob->value(j, i) = distribution(generator);
    }
  }
}

TEST(PartOfSpeechWordLayerTest, TestForwardCpu) {
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
  auto in = autoencoder::Blobs{&recurrent_input, &word_input};
  auto out = autoencoder::Blobs{&tag_output, &recurrent_output};
  layer.ForwardCpu(in, &out);
  for (auto i = 0; i < tag_output.width; ++i) {
    tag_output.difference(i) = i == 2;
  }
  for (auto i = 0; i < recurrent_output.width; ++i) {
    recurrent_output.difference(i) = 1.0f;
  }
  layer.BackwardCpu(out, &in);

  // Characterization tests.
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(0));
  EXPECT_FLOAT_EQ(9.7273235f, recurrent_input.difference(1));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(2));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(3));
  EXPECT_FLOAT_EQ(10.839175f, recurrent_input.difference(4));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(5));
  EXPECT_FLOAT_EQ(9.9717798f, recurrent_input.difference(6));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(7));
  EXPECT_FLOAT_EQ(10.31706f, recurrent_input.difference(8));
  EXPECT_FLOAT_EQ(8.5000582f, recurrent_input.difference(9));

  EXPECT_FLOAT_EQ(0.0f, word_input.difference(0));
  EXPECT_FLOAT_EQ(11.71076f, word_input.difference(1));
  EXPECT_FLOAT_EQ(9.2745285f, word_input.difference(2));
  EXPECT_FLOAT_EQ(0.0f, word_input.difference(3));
  EXPECT_FLOAT_EQ(0.0f, word_input.difference(4));
  EXPECT_FLOAT_EQ(0.0f, word_input.difference(5));
  EXPECT_FLOAT_EQ(8.0085802f, word_input.difference(6));
  EXPECT_FLOAT_EQ(9.6799927f, word_input.difference(7));
  EXPECT_FLOAT_EQ(0.0f, word_input.difference(8));
  EXPECT_FLOAT_EQ(8.7112932f, word_input.difference(9));

  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(0, 0));
  EXPECT_FLOAT_EQ(-0.0082464581f, classify_weights.difference(1, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(3, 0));
  EXPECT_FLOAT_EQ(0.010577375f, classify_weights.difference(4, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 0));
  EXPECT_FLOAT_EQ(-0.0019871078f, classify_weights.difference(6, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(7, 0));
  EXPECT_FLOAT_EQ(-0.008498692f, classify_weights.difference(8, 0));
  EXPECT_FLOAT_EQ(0.00034390931f, classify_weights.difference(9, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(0, 1));
  EXPECT_FLOAT_EQ(-0.024505166f, classify_weights.difference(1, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(3, 1));
  EXPECT_FLOAT_EQ(0.031431716f, classify_weights.difference(4, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 1));
  EXPECT_FLOAT_EQ(-0.0059048869f, classify_weights.difference(6, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(7, 1));
  EXPECT_FLOAT_EQ(-0.025254704f, classify_weights.difference(8, 1));
  EXPECT_FLOAT_EQ(0.0010219605f, classify_weights.difference(9, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(0, 2));
  EXPECT_FLOAT_EQ(0.073085397f, classify_weights.difference(1, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(3, 2));
  EXPECT_FLOAT_EQ(-0.093743473f, classify_weights.difference(4, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 2));
  EXPECT_FLOAT_EQ(0.017611021f, classify_weights.difference(6, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(7, 2));
  EXPECT_FLOAT_EQ(0.075320855f, classify_weights.difference(8, 2));
  EXPECT_FLOAT_EQ(-0.0030479445f, classify_weights.difference(9, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(0, 3));
  EXPECT_FLOAT_EQ(-0.021061296f, classify_weights.difference(1, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(3, 3));
  EXPECT_FLOAT_EQ(0.027014412f, classify_weights.difference(4, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 3));
  EXPECT_FLOAT_EQ(-0.0050750352f, classify_weights.difference(6, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(7, 3));
  EXPECT_FLOAT_EQ(-0.021705497f, classify_weights.difference(8, 3));
  EXPECT_FLOAT_EQ(0.00087833777f, classify_weights.difference(9, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(0, 4));
  EXPECT_FLOAT_EQ(-0.019272478f, classify_weights.difference(1, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(3, 4));
  EXPECT_FLOAT_EQ(0.024719972f, classify_weights.difference(4, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 4));
  EXPECT_FLOAT_EQ(-0.0046439925f, classify_weights.difference(6, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(7, 4));
  EXPECT_FLOAT_EQ(-0.019861965f, classify_weights.difference(8, 4));
  EXPECT_FLOAT_EQ(0.00080373709f, classify_weights.difference(9, 4));

  EXPECT_FLOAT_EQ(-0.0096809734f, classify_bias.difference(0));
  EXPECT_FLOAT_EQ(-0.028767969f, classify_bias.difference(1));
  EXPECT_FLOAT_EQ(0.085798986f, classify_bias.difference(2));
  EXPECT_FLOAT_EQ(-0.02472502f, classify_bias.difference(3));
  EXPECT_FLOAT_EQ(-0.022625027f, classify_bias.difference(4));

  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 0));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 0));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 0));
  EXPECT_FLOAT_EQ(0.20525908f, combine_weights.difference(6, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 0));
  EXPECT_FLOAT_EQ(0.8778758f, combine_weights.difference(8, 0));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 0));
  EXPECT_FLOAT_EQ(1.120111f, combine_weights.difference(11, 0));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 0));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 0));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 0));
  EXPECT_FLOAT_EQ(-0.39592981f, combine_weights.difference(19, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 1));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 1));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 1));
  EXPECT_FLOAT_EQ(0.20525908f, combine_weights.difference(6, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 1));
  EXPECT_FLOAT_EQ(0.8778758f, combine_weights.difference(8, 1));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 1));
  EXPECT_FLOAT_EQ(1.120111f, combine_weights.difference(11, 1));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 1));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 1));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 1));
  EXPECT_FLOAT_EQ(-0.39592981f, combine_weights.difference(19, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 2));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 2));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 2));
  EXPECT_FLOAT_EQ(0.20525908f, combine_weights.difference(6, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 2));
  EXPECT_FLOAT_EQ(0.8778758f, combine_weights.difference(8, 2));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 2));
  EXPECT_FLOAT_EQ(1.120111f, combine_weights.difference(11, 2));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 2));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 2));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 2));
  EXPECT_FLOAT_EQ(-0.39592981f, combine_weights.difference(19, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 3));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 3));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 3));
  EXPECT_FLOAT_EQ(0.20525908f, combine_weights.difference(6, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 3));
  EXPECT_FLOAT_EQ(0.8778758f, combine_weights.difference(8, 3));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 3));
  EXPECT_FLOAT_EQ(1.120111f, combine_weights.difference(11, 3));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 3));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 3));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 3));
  EXPECT_FLOAT_EQ(-0.39592981f, combine_weights.difference(19, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 4));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 4));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 4));
  EXPECT_FLOAT_EQ(0.20525908f, combine_weights.difference(6, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 4));
  EXPECT_FLOAT_EQ(0.8778758f, combine_weights.difference(8, 4));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 4));
  EXPECT_FLOAT_EQ(1.120111f, combine_weights.difference(11, 4));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 4));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 4));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 4));
  EXPECT_FLOAT_EQ(-0.39592981f, combine_weights.difference(19, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 5));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 5));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 5));
  EXPECT_FLOAT_EQ(0.20525908f, combine_weights.difference(6, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 5));
  EXPECT_FLOAT_EQ(0.8778758f, combine_weights.difference(8, 5));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 5));
  EXPECT_FLOAT_EQ(1.120111f, combine_weights.difference(11, 5));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 5));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 5));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 5));
  EXPECT_FLOAT_EQ(-0.39592981f, combine_weights.difference(19, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 6));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 6));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 6));
  EXPECT_FLOAT_EQ(0.20525908f, combine_weights.difference(6, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 6));
  EXPECT_FLOAT_EQ(0.8778758f, combine_weights.difference(8, 6));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 6));
  EXPECT_FLOAT_EQ(1.120111f, combine_weights.difference(11, 6));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 6));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 6));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 6));
  EXPECT_FLOAT_EQ(-0.39592981f, combine_weights.difference(19, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 7));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 7));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 7));
  EXPECT_FLOAT_EQ(0.20525908f, combine_weights.difference(6, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 7));
  EXPECT_FLOAT_EQ(0.8778758f, combine_weights.difference(8, 7));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 7));
  EXPECT_FLOAT_EQ(1.120111f, combine_weights.difference(11, 7));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 7));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 7));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 7));
  EXPECT_FLOAT_EQ(-0.39592981f, combine_weights.difference(19, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 8));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 8));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 8));
  EXPECT_FLOAT_EQ(0.20525908f, combine_weights.difference(6, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 8));
  EXPECT_FLOAT_EQ(0.8778758f, combine_weights.difference(8, 8));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 8));
  EXPECT_FLOAT_EQ(1.120111f, combine_weights.difference(11, 8));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 8));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 8));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 8));
  EXPECT_FLOAT_EQ(-0.39592981f, combine_weights.difference(19, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 9));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 9));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 9));
  EXPECT_FLOAT_EQ(0.20525908f, combine_weights.difference(6, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 9));
  EXPECT_FLOAT_EQ(0.8778758f, combine_weights.difference(8, 9));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 9));
  EXPECT_FLOAT_EQ(1.120111f, combine_weights.difference(11, 9));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 9));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 9));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 9));
  EXPECT_FLOAT_EQ(-0.39592981f, combine_weights.difference(19, 9));

  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(0));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(1));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(2));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(3));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(4));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(5));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(6));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(7));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(8));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(9));
}
