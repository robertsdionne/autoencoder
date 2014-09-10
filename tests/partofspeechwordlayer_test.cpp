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
      0.5f, classify_weights, classify_bias, combine_weights, combine_bias, generator);
  auto tag_output = autoencoder::Blob(5);
  auto recurrent_output = autoencoder::Blob(10);
  auto out = autoencoder::Blobs{&tag_output, &recurrent_output};
  layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, {&recurrent_input, &word_input}, &out);

  auto sum = 0.0f;
  for (auto i = 0; i < tag_output.width; ++i) {
    sum += tag_output.value(i);
  }
  EXPECT_FLOAT_EQ(1.0f, sum);

  // Characterization tests.
  EXPECT_FLOAT_EQ(0.12880097f, tag_output.value(0));
  EXPECT_FLOAT_EQ(0.31371129f, tag_output.value(1));
  EXPECT_FLOAT_EQ(0.059748095f, tag_output.value(2));
  EXPECT_FLOAT_EQ(0.33044919f, tag_output.value(3));
  EXPECT_FLOAT_EQ(0.16729045f, tag_output.value(4));

  EXPECT_FLOAT_EQ(1.7310448f, recurrent_output.value(0));
  EXPECT_FLOAT_EQ(0.89490986f, recurrent_output.value(1));
  EXPECT_FLOAT_EQ(1.4617033f, recurrent_output.value(2));
  EXPECT_FLOAT_EQ(2.0389292f, recurrent_output.value(3));
  EXPECT_FLOAT_EQ(0.86021233f, recurrent_output.value(4));
  EXPECT_FLOAT_EQ(0.24303776f, recurrent_output.value(5));
  EXPECT_FLOAT_EQ(0.2527028f, recurrent_output.value(6));
  EXPECT_FLOAT_EQ(0.0f, recurrent_output.value(7));
  EXPECT_FLOAT_EQ(0.26613188f, recurrent_output.value(8));
  EXPECT_FLOAT_EQ(0.78978527f, recurrent_output.value(9));
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
      0.5f, classify_weights, classify_bias, combine_weights, combine_bias, generator);
  auto tag_output = autoencoder::Blob(5);
  auto recurrent_output = autoencoder::Blob(10);
  auto in = autoencoder::Blobs{&recurrent_input, &word_input};
  auto out = autoencoder::Blobs{&tag_output, &recurrent_output};
  layer.ForwardCpu(autoencoder::Layer::Mode::kTrain, in, &out);
  for (auto i = 0; i < tag_output.width; ++i) {
    tag_output.difference(i) = i == 2;
  }
  for (auto i = 0; i < recurrent_output.width; ++i) {
    recurrent_output.difference(i) = 1.0f;
  }
  layer.BackwardCpu(out, &in);

  // Characterization tests.
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(0));
  EXPECT_FLOAT_EQ(8.0420856f, recurrent_input.difference(1));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(2));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(3));
  EXPECT_FLOAT_EQ(9.4588242f, recurrent_input.difference(4));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(5));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(6));
  EXPECT_FLOAT_EQ(10.232465f, recurrent_input.difference(7));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(8));
  EXPECT_FLOAT_EQ(8.1536922f, recurrent_input.difference(9));

  EXPECT_FLOAT_EQ(9.0055218f, word_input.difference(0));
  EXPECT_FLOAT_EQ(0.0f, word_input.difference(1));
  EXPECT_FLOAT_EQ(9.1566525f, word_input.difference(2));
  EXPECT_FLOAT_EQ(0.0f, word_input.difference(3));
  EXPECT_FLOAT_EQ(0.0f, word_input.difference(4));
  EXPECT_FLOAT_EQ(0.0f, word_input.difference(5));
  EXPECT_FLOAT_EQ(6.9662609f, word_input.difference(6));
  EXPECT_FLOAT_EQ(8.5228901f, word_input.difference(7));
  EXPECT_FLOAT_EQ(0.0f, word_input.difference(8));
  EXPECT_FLOAT_EQ(0.0f, word_input.difference(9));

  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(0, 0));
  EXPECT_FLOAT_EQ(-0.0065552858f, classify_weights.difference(1, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(3, 0));
  EXPECT_FLOAT_EQ(0.0084081814f, classify_weights.difference(4, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(6, 0));
  EXPECT_FLOAT_EQ(-0.0067459834f, classify_weights.difference(7, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(8, 0));
  EXPECT_FLOAT_EQ(0.00027338087f, classify_weights.difference(9, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(0, 1));
  EXPECT_FLOAT_EQ(-0.015966238f, classify_weights.difference(1, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(3, 1));
  EXPECT_FLOAT_EQ(0.020479204f, classify_weights.difference(4, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(6, 1));
  EXPECT_FLOAT_EQ(-0.016430708f, classify_weights.difference(7, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(8, 1));
  EXPECT_FLOAT_EQ(0.00066585414f, classify_weights.difference(9, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(0, 2));
  EXPECT_FLOAT_EQ(0.047853827f, classify_weights.difference(1, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(3, 2));
  EXPECT_FLOAT_EQ(-0.061380036f, classify_weights.difference(4, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(6, 2));
  EXPECT_FLOAT_EQ(0.049245927f, classify_weights.difference(7, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(8, 2));
  EXPECT_FLOAT_EQ(-0.0019956904f, classify_weights.difference(9, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(0, 3));
  EXPECT_FLOAT_EQ(-0.01681811f, classify_weights.difference(1, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(3, 3));
  EXPECT_FLOAT_EQ(0.021571862f, classify_weights.difference(4, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(6, 3));
  EXPECT_FLOAT_EQ(-0.01730736f, classify_weights.difference(7, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(8, 3));
  EXPECT_FLOAT_EQ(0.00070138043f, classify_weights.difference(9, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(0, 4));
  EXPECT_FLOAT_EQ(-0.0085141957f, classify_weights.difference(1, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(3, 4));
  EXPECT_FLOAT_EQ(0.01092079f, classify_weights.difference(4, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(6, 4));
  EXPECT_FLOAT_EQ(-0.0087618791f, classify_weights.difference(7, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(8, 4));
  EXPECT_FLOAT_EQ(0.000355075f, classify_weights.difference(9, 4));

  EXPECT_FLOAT_EQ(-0.0076956125f, classify_bias.difference(0));
  EXPECT_FLOAT_EQ(-0.018743651f, classify_bias.difference(1));
  EXPECT_FLOAT_EQ(0.056178257f, classify_bias.difference(2));
  EXPECT_FLOAT_EQ(-0.019743709f, classify_bias.difference(3));
  EXPECT_FLOAT_EQ(-0.0099952854f, classify_bias.difference(4));

  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 0));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 0));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 0));
  EXPECT_FLOAT_EQ(0.87660122f, combine_weights.difference(7, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 0));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 0));
  EXPECT_FLOAT_EQ(-0.30757415f, combine_weights.difference(10, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 0));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 0));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 0));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 0));
  EXPECT_FLOAT_EQ(-0.0f, combine_weights.difference(19, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 1));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 1));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 1));
  EXPECT_FLOAT_EQ(0.87660122f, combine_weights.difference(7, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 1));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 1));
  EXPECT_FLOAT_EQ(-0.30757415f, combine_weights.difference(10, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 1));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 1));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 1));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 2));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 2));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 2));
  EXPECT_FLOAT_EQ(0.87660122f, combine_weights.difference(7, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 2));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 2));
  EXPECT_FLOAT_EQ(-0.30757415f, combine_weights.difference(10, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 2));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 2));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 2));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 3));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 3));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 3));
  EXPECT_FLOAT_EQ(0.87660122f, combine_weights.difference(7, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 3));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 3));
  EXPECT_FLOAT_EQ(-0.30757415f, combine_weights.difference(10, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 3));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 3));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 3));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 4));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 4));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 4));
  EXPECT_FLOAT_EQ(0.87660122f, combine_weights.difference(7, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 4));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 4));
  EXPECT_FLOAT_EQ(-0.30757415f, combine_weights.difference(10, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 4));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 4));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 4));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 5));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 5));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 5));
  EXPECT_FLOAT_EQ(0.87660122f, combine_weights.difference(7, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 5));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 5));
  EXPECT_FLOAT_EQ(-0.30757415f, combine_weights.difference(10, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 5));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 5));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 5));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 6));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 6));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 6));
  EXPECT_FLOAT_EQ(0.87660122f, combine_weights.difference(7, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 6));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 6));
  EXPECT_FLOAT_EQ(-0.30757415f, combine_weights.difference(10, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 6));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 6));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 6));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(1, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(9, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(12, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(16, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(17, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 8));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 8));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 8));
  EXPECT_FLOAT_EQ(0.87660122f, combine_weights.difference(7, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 8));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 8));
  EXPECT_FLOAT_EQ(-0.30757415f, combine_weights.difference(10, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 8));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 8));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 8));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 9));
  EXPECT_FLOAT_EQ(0.85182118f, combine_weights.difference(1, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 9));
  EXPECT_FLOAT_EQ(-1.0925941f, combine_weights.difference(4, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 9));
  EXPECT_FLOAT_EQ(0.87660122f, combine_weights.difference(7, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 9));
  EXPECT_FLOAT_EQ(-0.035524249f, combine_weights.difference(9, 9));
  EXPECT_FLOAT_EQ(-0.30757415f, combine_weights.difference(10, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 9));
  EXPECT_FLOAT_EQ(1.9230568f, combine_weights.difference(12, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 9));
  EXPECT_FLOAT_EQ(-0.076272368f, combine_weights.difference(16, 9));
  EXPECT_FLOAT_EQ(-1.440197f, combine_weights.difference(17, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 9));

  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(0));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(1));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(2));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(3));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(4));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(5));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(6));
  EXPECT_FLOAT_EQ(0.0f, combine_bias.difference(7));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(8));
  EXPECT_FLOAT_EQ(1.0f, combine_bias.difference(9));
}
