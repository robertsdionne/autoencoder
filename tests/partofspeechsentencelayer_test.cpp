#include <gtest/gtest.h>
#include <random>

#include "blob.hpp"
#include "partofspeechsentencelayer.hpp"

constexpr auto kRandomSeed = 123;

template<typename Distribution, typename Generator>
void InitializeBlob(Distribution &distribution, Generator &generator, autoencoder::Blob *blob) {
  for (auto i = 0; i < blob->height; ++i) {
    for (auto j = 0; j < blob->width; ++j) {
      blob->value(j, i) = distribution(generator);
    }
  }
}

TEST(PartOfSpeechSentenceLayerTest, TestForwardCpu) {
  std::mt19937 generator(kRandomSeed);
  std::uniform_real_distribution<float> uniform;
  std::uniform_real_distribution<float> uniform_symmetric(-1.0f, 1.0f);

  auto recurrent_input = autoencoder::Blob(10);
  auto word_input0 = autoencoder::Blob(10);
  auto word_input1 = autoencoder::Blob(10);
  auto word_input2 = autoencoder::Blob(10);
  auto word_input3 = autoencoder::Blob(10);
  auto word_input4 = autoencoder::Blob(10);
  auto classify_weights = autoencoder::Blob(10, 5);
  auto classify_bias = autoencoder::Blob(5);
  auto combine_weights = autoencoder::Blob(20, 10);
  auto combine_bias = autoencoder::Blob(10);

  InitializeBlob(uniform_symmetric, generator, &recurrent_input);
  InitializeBlob(uniform_symmetric, generator, &word_input0);
  InitializeBlob(uniform, generator, &classify_weights);
  InitializeBlob(uniform, generator, &classify_bias);
  InitializeBlob(uniform, generator, &combine_weights);
  InitializeBlob(uniform, generator, &combine_bias);
  InitializeBlob(uniform_symmetric, generator, &word_input1);
  InitializeBlob(uniform_symmetric, generator, &word_input2);
  InitializeBlob(uniform_symmetric, generator, &word_input3);
  InitializeBlob(uniform_symmetric, generator, &word_input4);

  auto layer = autoencoder::PartOfSpeechSentenceLayer(
      0.5f, classify_weights, classify_bias, combine_weights, combine_bias, generator);
  auto tag_output0 = autoencoder::Blob(5);
  auto tag_output1 = autoencoder::Blob(5);
  auto tag_output2 = autoencoder::Blob(5);
  auto tag_output3 = autoencoder::Blob(5);
  auto tag_output4 = autoencoder::Blob(5);
  auto recurrent_output = autoencoder::Blob(10);
  auto in = autoencoder::Blobs{
    &recurrent_input, &word_input0, &word_input1, &word_input2, &word_input3, &word_input4
  };
  auto out = autoencoder::Blobs{
    &tag_output0, &tag_output1, &tag_output2, &tag_output3, &tag_output4, &recurrent_output
  };
  layer.ForwardCpu(in, &out);

  for (auto &tag_output : {tag_output0, tag_output1, tag_output2, tag_output3, tag_output4}) {
    auto sum = 0.0f;
    for (auto i = 0; i < tag_output.width; ++i) {
      sum += tag_output.value(i);
    }
    EXPECT_FLOAT_EQ(1.0f, sum);
  }

  // Characterization tests.
  EXPECT_FLOAT_EQ(0.11221863f, tag_output0.value(0));
  EXPECT_FLOAT_EQ(0.29665101f, tag_output0.value(1));
  EXPECT_FLOAT_EQ(0.20036446f, tag_output0.value(2));
  EXPECT_FLOAT_EQ(0.20704912f, tag_output0.value(3));
  EXPECT_FLOAT_EQ(0.18371686f, tag_output0.value(4));

  EXPECT_FLOAT_EQ(0.065541506f, tag_output1.value(0));
  EXPECT_FLOAT_EQ(0.11427262f, tag_output1.value(1));
  EXPECT_FLOAT_EQ(0.011574534f, tag_output1.value(2));
  EXPECT_FLOAT_EQ(0.8080343f, tag_output1.value(3));
  EXPECT_FLOAT_EQ(0.00057711347f, tag_output1.value(4));

  EXPECT_FLOAT_EQ(0.00033127228f, tag_output2.value(0));
  EXPECT_FLOAT_EQ(0.10332016f, tag_output2.value(1));
  EXPECT_FLOAT_EQ(0.4202475f, tag_output2.value(2));
  EXPECT_FLOAT_EQ(0.4761011f, tag_output2.value(3));
  EXPECT_FLOAT_EQ(2.0404652e-08f, tag_output2.value(4));

  EXPECT_FLOAT_EQ(9.251499e-21f, tag_output3.value(0));
  EXPECT_FLOAT_EQ(1.6191652e-30f, tag_output3.value(1));
  EXPECT_FLOAT_EQ(1.0f, tag_output3.value(2));
  EXPECT_FLOAT_EQ(3.4942835e-09f, tag_output3.value(3));
  EXPECT_FLOAT_EQ(0.0f, tag_output3.value(4));

  EXPECT_FLOAT_EQ(0.0f, tag_output4.value(0));
  EXPECT_FLOAT_EQ(0.0f, tag_output4.value(1));
  EXPECT_FLOAT_EQ(1.0f, tag_output4.value(2));
  EXPECT_FLOAT_EQ(0.0f, tag_output4.value(3));
  EXPECT_FLOAT_EQ(0.0f, tag_output4.value(4));

  EXPECT_FLOAT_EQ(1210.7053f, recurrent_output.value(0));
  EXPECT_FLOAT_EQ(1471.5319f, recurrent_output.value(1));
  EXPECT_FLOAT_EQ(1533.4453f, recurrent_output.value(2));
  EXPECT_FLOAT_EQ(817.5282f, recurrent_output.value(3));
  EXPECT_FLOAT_EQ(1165.9307f, recurrent_output.value(4));
  EXPECT_FLOAT_EQ(957.3028f, recurrent_output.value(5));
  EXPECT_FLOAT_EQ(1375.8092f, recurrent_output.value(6));
  EXPECT_FLOAT_EQ(1670.3793f, recurrent_output.value(7));
  EXPECT_FLOAT_EQ(1105.9539f, recurrent_output.value(8));
  EXPECT_FLOAT_EQ(1070.3689f, recurrent_output.value(9));
}

TEST(PartOfSpeechSentenceLayerTest, TestBackwardCpu) {
  std::mt19937 generator(kRandomSeed);
  std::uniform_real_distribution<float> uniform;
  std::uniform_real_distribution<float> uniform_symmetric(-1.0f, 1.0f);

  auto recurrent_input = autoencoder::Blob(10);
  auto word_input0 = autoencoder::Blob(10);
  auto word_input1 = autoencoder::Blob(10);
  auto word_input2 = autoencoder::Blob(10);
  auto word_input3 = autoencoder::Blob(10);
  auto word_input4 = autoencoder::Blob(10);
  auto classify_weights = autoencoder::Blob(10, 5);
  auto classify_bias = autoencoder::Blob(5);
  auto combine_weights = autoencoder::Blob(20, 10);
  auto combine_bias = autoencoder::Blob(10);

  InitializeBlob(uniform_symmetric, generator, &recurrent_input);
  InitializeBlob(uniform_symmetric, generator, &word_input0);
  InitializeBlob(uniform, generator, &classify_weights);
  InitializeBlob(uniform, generator, &classify_bias);
  InitializeBlob(uniform, generator, &combine_weights);
  InitializeBlob(uniform, generator, &combine_bias);
  InitializeBlob(uniform_symmetric, generator, &word_input1);
  InitializeBlob(uniform_symmetric, generator, &word_input2);
  InitializeBlob(uniform_symmetric, generator, &word_input3);
  InitializeBlob(uniform_symmetric, generator, &word_input4);

  auto layer = autoencoder::PartOfSpeechSentenceLayer(
      0.5f, classify_weights, classify_bias, combine_weights, combine_bias, generator);
  auto tag_output0 = autoencoder::Blob(5);
  auto tag_output1 = autoencoder::Blob(5);
  auto tag_output2 = autoencoder::Blob(5);
  auto tag_output3 = autoencoder::Blob(5);
  auto tag_output4 = autoencoder::Blob(5);
  auto recurrent_output = autoencoder::Blob(10);
  auto in = autoencoder::Blobs{
    &recurrent_input, &word_input0, &word_input1, &word_input2, &word_input3, &word_input4
  };
  auto out = autoencoder::Blobs{
    &tag_output0, &tag_output1, &tag_output2, &tag_output3, &tag_output4, &recurrent_output
  };
  layer.ForwardCpu(in, &out);

  {
    auto i = 0;
    for (auto tag_output : {&tag_output0, &tag_output1, &tag_output2, &tag_output3, &tag_output4}) {
      for (auto j = 0; j < tag_output->width; ++j) {
        tag_output->difference(j) = i == j;
      }
    }
  }
  for (auto i = 0; i < recurrent_output.width; ++i) {
    recurrent_output.difference(i) = 1.0f;
  }
  layer.BackwardCpu(out, &in);

  // Characterization tests.
  EXPECT_FLOAT_EQ(-0.099233434f, recurrent_input.difference(0));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(1));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(2));
  EXPECT_FLOAT_EQ(-0.053486858f, recurrent_input.difference(3));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(4));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(5));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(6));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(7));
  EXPECT_FLOAT_EQ(-0.10154264f, recurrent_input.difference(8));
  EXPECT_FLOAT_EQ(0.0f, recurrent_input.difference(9));

  EXPECT_FLOAT_EQ(0.0f, word_input0.difference(0));
  EXPECT_FLOAT_EQ(0.0f, word_input0.difference(1));
  EXPECT_FLOAT_EQ(-0.031720333f, word_input0.difference(2));
  EXPECT_FLOAT_EQ(-0.053825542f, word_input0.difference(3));
  EXPECT_FLOAT_EQ(-0.062813051f, word_input0.difference(4));
  EXPECT_FLOAT_EQ(-0.04747666f, word_input0.difference(5));
  EXPECT_FLOAT_EQ(-0.0410267f, word_input0.difference(6));
  EXPECT_FLOAT_EQ(0.0f, word_input0.difference(7));
  EXPECT_FLOAT_EQ(-0.0476528f, word_input0.difference(8));
  EXPECT_FLOAT_EQ(-0.044908211f, word_input0.difference(9));

  EXPECT_FLOAT_EQ(0.0f, word_input1.difference(0));
  EXPECT_FLOAT_EQ(-0.0001589093f, word_input1.difference(1));
  EXPECT_FLOAT_EQ(0.0f, word_input1.difference(2));
  EXPECT_FLOAT_EQ(-0.00015711761f, word_input1.difference(3));
  EXPECT_FLOAT_EQ(0.0f, word_input1.difference(4));
  EXPECT_FLOAT_EQ(-0.00042619626f, word_input1.difference(5));
  EXPECT_FLOAT_EQ(-0.00014309862f, word_input1.difference(6));
  EXPECT_FLOAT_EQ(0.0f, word_input1.difference(7));
  EXPECT_FLOAT_EQ(-0.00022382228f, word_input1.difference(8));
  EXPECT_FLOAT_EQ(0.0f, word_input1.difference(9));

  EXPECT_FLOAT_EQ(0.0f, word_input2.difference(0));
  EXPECT_FLOAT_EQ(-1.4395783e-20f, word_input2.difference(1));
  EXPECT_FLOAT_EQ(-2.8019722e-20f, word_input2.difference(2));
  EXPECT_FLOAT_EQ(-2.1472097e-20f, word_input2.difference(3));
  EXPECT_FLOAT_EQ(-6.1074949e-21f, word_input2.difference(4));
  EXPECT_FLOAT_EQ(0.0f, word_input2.difference(5));
  EXPECT_FLOAT_EQ(-2.3153244e-20f, word_input2.difference(6));
  EXPECT_FLOAT_EQ(-1.9980178e-20f, word_input2.difference(7));
  EXPECT_FLOAT_EQ(0.0f, word_input2.difference(8));
  EXPECT_FLOAT_EQ(-1.9609031e-20f, word_input2.difference(9));

  EXPECT_FLOAT_EQ(0.0f, word_input3.difference(0));
  EXPECT_FLOAT_EQ(0.0f, word_input3.difference(1));
  EXPECT_FLOAT_EQ(0.0f, word_input3.difference(2));
  EXPECT_FLOAT_EQ(0.0f, word_input3.difference(3));
  EXPECT_FLOAT_EQ(0.0f, word_input3.difference(4));
  EXPECT_FLOAT_EQ(0.0f, word_input3.difference(5));
  EXPECT_FLOAT_EQ(0.0f, word_input3.difference(6));
  EXPECT_FLOAT_EQ(0.0f, word_input3.difference(7));
  EXPECT_FLOAT_EQ(0.0f, word_input3.difference(8));
  EXPECT_FLOAT_EQ(0.0f, word_input3.difference(9));

  EXPECT_FLOAT_EQ(0.0f, word_input4.difference(0));
  EXPECT_FLOAT_EQ(0.0f, word_input4.difference(1));
  EXPECT_FLOAT_EQ(0.0f, word_input4.difference(2));
  EXPECT_FLOAT_EQ(0.0f, word_input4.difference(3));
  EXPECT_FLOAT_EQ(0.0f, word_input4.difference(4));
  EXPECT_FLOAT_EQ(0.0f, word_input4.difference(5));
  EXPECT_FLOAT_EQ(0.0f, word_input4.difference(6));
  EXPECT_FLOAT_EQ(0.0f, word_input4.difference(7));
  EXPECT_FLOAT_EQ(0.0f, word_input4.difference(8));
  EXPECT_FLOAT_EQ(0.0f, word_input4.difference(9));

  EXPECT_FLOAT_EQ(0.07829345f, classify_weights.difference(0, 0));
  EXPECT_FLOAT_EQ(0.16914907f, classify_weights.difference(1, 0));
  EXPECT_FLOAT_EQ(5.2011338e-19f, classify_weights.difference(2, 0));
  EXPECT_FLOAT_EQ(0.32273859f, classify_weights.difference(3, 0));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(4, 0));
  EXPECT_FLOAT_EQ(4.3870129e-19f, classify_weights.difference(5, 0));
  EXPECT_FLOAT_EQ(0.0039053995f, classify_weights.difference(6, 0));
  EXPECT_FLOAT_EQ(0.19815832f, classify_weights.difference(7, 0));
  EXPECT_FLOAT_EQ(0.087458916f, classify_weights.difference(8, 0));
  EXPECT_FLOAT_EQ(0.27995023f, classify_weights.difference(9, 0));
  EXPECT_FLOAT_EQ(-0.026161654f, classify_weights.difference(0, 1));
  EXPECT_FLOAT_EQ(-0.020684822f, classify_weights.difference(1, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 1));
  EXPECT_FLOAT_EQ(-0.033292983f, classify_weights.difference(3, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(4, 1));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 1));
  EXPECT_FLOAT_EQ(-0.00040364024f, classify_weights.difference(6, 1));
  EXPECT_FLOAT_EQ(-0.024232293f, classify_weights.difference(7, 1));
  EXPECT_FLOAT_EQ(-0.029224282f, classify_weights.difference(8, 1));
  EXPECT_FLOAT_EQ(-0.034124978f, classify_weights.difference(9, 1));
  EXPECT_FLOAT_EQ(-0.017670143f, classify_weights.difference(0, 2));
  EXPECT_FLOAT_EQ(-0.0020951401f, classify_weights.difference(1, 2));
  EXPECT_FLOAT_EQ(-5.2011338e-19f, classify_weights.difference(2, 2));
  EXPECT_FLOAT_EQ(-0.00082491105f, classify_weights.difference(3, 2));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(4, 2));
  EXPECT_FLOAT_EQ(-4.3870129e-19f, classify_weights.difference(5, 2));
  EXPECT_FLOAT_EQ(-0.0016417784f, classify_weights.difference(6, 2));
  EXPECT_FLOAT_EQ(-0.002454459f, classify_weights.difference(7, 2));
  EXPECT_FLOAT_EQ(-0.019738708f, classify_weights.difference(8, 2));
  EXPECT_FLOAT_EQ(-0.0058260038f, classify_weights.difference(9, 2));
  EXPECT_FLOAT_EQ(-0.018259663f, classify_weights.difference(0, 3));
  EXPECT_FLOAT_EQ(-0.14626466f, classify_weights.difference(1, 3));
  EXPECT_FLOAT_EQ(-1.8174236e-27f, classify_weights.difference(2, 3));
  EXPECT_FLOAT_EQ(-0.29430687f, classify_weights.difference(3, 3));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(4, 3));
  EXPECT_FLOAT_EQ(-1.5329467e-27f, classify_weights.difference(5, 3));
  EXPECT_FLOAT_EQ(-0.0018599812f, classify_weights.difference(6, 3));
  EXPECT_FLOAT_EQ(-0.1713492f, classify_weights.difference(7, 3));
  EXPECT_FLOAT_EQ(-0.02039724f, classify_weights.difference(8, 3));
  EXPECT_FLOAT_EQ(-0.23982993f, classify_weights.difference(9, 3));
  EXPECT_FLOAT_EQ(-0.016201992f, classify_weights.difference(0, 4));
  EXPECT_FLOAT_EQ(-0.00010446501f, classify_weights.difference(1, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(2, 4));
  EXPECT_FLOAT_EQ(0.0056861783f, classify_weights.difference(3, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(4, 4));
  EXPECT_FLOAT_EQ(0.0f, classify_weights.difference(5, 4));
  EXPECT_FLOAT_EQ(-7.9714728e-11f, classify_weights.difference(6, 4));
  EXPECT_FLOAT_EQ(-0.00012238086f, classify_weights.difference(7, 4));
  EXPECT_FLOAT_EQ(-0.018098686f, classify_weights.difference(8, 4));
  EXPECT_FLOAT_EQ(-0.00016932498f, classify_weights.difference(9, 4));

  EXPECT_FLOAT_EQ(0.16120259f, classify_bias.difference(0));
  EXPECT_FLOAT_EQ(-0.040813595f, classify_bias.difference(1));
  EXPECT_FLOAT_EQ(-0.023382453f, classify_bias.difference(2));
  EXPECT_FLOAT_EQ(-0.076352268f, classify_bias.difference(3));
  EXPECT_FLOAT_EQ(-0.020654278f, classify_bias.difference(4));

  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(1, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(9, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(12, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(16, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(17, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 0));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 0));
  EXPECT_FLOAT_EQ(-0.0020615913f, combine_weights.difference(0, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(1, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 1));
  EXPECT_FLOAT_EQ(0.00075056904f, combine_weights.difference(3, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 1));
  EXPECT_FLOAT_EQ(-0.0023029326f, combine_weights.difference(8, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(9, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 1));
  EXPECT_FLOAT_EQ(-0.0050447569f, combine_weights.difference(12, 1));
  EXPECT_FLOAT_EQ(0.00093468872f, combine_weights.difference(13, 1));
  EXPECT_FLOAT_EQ(-0.0019394559f, combine_weights.difference(14, 1));
  EXPECT_FLOAT_EQ(-0.00083624822f, combine_weights.difference(15, 1));
  EXPECT_FLOAT_EQ(0.00020008539f, combine_weights.difference(16, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(17, 1));
  EXPECT_FLOAT_EQ(0.0011320327f, combine_weights.difference(18, 1));
  EXPECT_FLOAT_EQ(0.0010386431f, combine_weights.difference(19, 1));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(1, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 2));
  EXPECT_FLOAT_EQ(-4.7931733e-20f, combine_weights.difference(3, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 2));
  EXPECT_FLOAT_EQ(-2.6268091e-20f, combine_weights.difference(6, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 2));
  EXPECT_FLOAT_EQ(-3.8880119e-20f, combine_weights.difference(9, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 2));
  EXPECT_FLOAT_EQ(-1.859436e-21f, combine_weights.difference(11, 2));
  EXPECT_FLOAT_EQ(2.1753721e-21f, combine_weights.difference(12, 2));
  EXPECT_FLOAT_EQ(-3.0225646e-21f, combine_weights.difference(13, 2));
  EXPECT_FLOAT_EQ(1.0556944e-21f, combine_weights.difference(14, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 2));
  EXPECT_FLOAT_EQ(-2.7387104e-21f, combine_weights.difference(16, 2));
  EXPECT_FLOAT_EQ(-2.503614e-21f, combine_weights.difference(17, 2));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 2));
  EXPECT_FLOAT_EQ(1.9018999e-21f, combine_weights.difference(19, 2));
  EXPECT_FLOAT_EQ(-0.0058818921f, combine_weights.difference(0, 3));
  EXPECT_FLOAT_EQ(-0.0001943426f, combine_weights.difference(1, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 3));
  EXPECT_FLOAT_EQ(0.0017460656f, combine_weights.difference(3, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 3));
  EXPECT_FLOAT_EQ(-3.6644153e-20f, combine_weights.difference(6, 3));
  EXPECT_FLOAT_EQ(-0.00022767257f, combine_weights.difference(7, 3));
  EXPECT_FLOAT_EQ(-0.0065704589f, combine_weights.difference(8, 3));
  EXPECT_FLOAT_EQ(-0.00031500534f, combine_weights.difference(9, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 3));
  EXPECT_FLOAT_EQ(-1.942261e-05f, combine_weights.difference(11, 3));
  EXPECT_FLOAT_EQ(-0.014393114f, combine_weights.difference(12, 3));
  EXPECT_FLOAT_EQ(0.0026787324f, combine_weights.difference(13, 3));
  EXPECT_FLOAT_EQ(-0.0055334298f, combine_weights.difference(14, 3));
  EXPECT_FLOAT_EQ(-0.0024572467f, combine_weights.difference(15, 3));
  EXPECT_FLOAT_EQ(0.00057821465f, combine_weights.difference(16, 3));
  EXPECT_FLOAT_EQ(-3.4925573e-21f, combine_weights.difference(17, 3));
  EXPECT_FLOAT_EQ(0.00322713f, combine_weights.difference(18, 3));
  EXPECT_FLOAT_EQ(0.0029633355f, combine_weights.difference(19, 3));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(1, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(3, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(6, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(9, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(11, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(12, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(13, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(14, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(16, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(17, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(19, 4));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(1, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 5));
  EXPECT_FLOAT_EQ(-1.9730457e-19f, combine_weights.difference(3, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 5));
  EXPECT_FLOAT_EQ(-1.0812909e-19f, combine_weights.difference(6, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 5));
  EXPECT_FLOAT_EQ(-1.6004482e-19f, combine_weights.difference(9, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 5));
  EXPECT_FLOAT_EQ(-7.6541203e-21f, combine_weights.difference(11, 5));
  EXPECT_FLOAT_EQ(8.9546292e-21f, combine_weights.difference(12, 5));
  EXPECT_FLOAT_EQ(-1.2441984e-20f, combine_weights.difference(13, 5));
  EXPECT_FLOAT_EQ(4.3456253e-21f, combine_weights.difference(14, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 5));
  EXPECT_FLOAT_EQ(-1.1273536e-20f, combine_weights.difference(16, 5));
  EXPECT_FLOAT_EQ(-1.0305793e-20f, combine_weights.difference(17, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 5));
  EXPECT_FLOAT_EQ(7.8289171e-21f, combine_weights.difference(19, 5));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 6));
  EXPECT_FLOAT_EQ(-0.00071707071f, combine_weights.difference(1, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 6));
  EXPECT_FLOAT_EQ(-0.0014588085f, combine_weights.difference(3, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 6));
  EXPECT_FLOAT_EQ(-1.2030683e-19f, combine_weights.difference(6, 6));
  EXPECT_FLOAT_EQ(-0.00084004912f, combine_weights.difference(7, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 6));
  EXPECT_FLOAT_EQ(-0.001162283f, combine_weights.difference(9, 6));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 6));
  EXPECT_FLOAT_EQ(-7.1664086e-05f, combine_weights.difference(11, 6));
  EXPECT_FLOAT_EQ(9.9631191e-21f, combine_weights.difference(12, 6));
  EXPECT_FLOAT_EQ(4.4229902e-05f, combine_weights.difference(13, 6));
  EXPECT_FLOAT_EQ(4.8350391e-21f, combine_weights.difference(14, 6));
  EXPECT_FLOAT_EQ(-0.00026330023f, combine_weights.difference(15, 6));
  EXPECT_FLOAT_EQ(2.7135469e-05f, combine_weights.difference(16, 6));
  EXPECT_FLOAT_EQ(-1.1466454e-20f, combine_weights.difference(17, 6));
  EXPECT_FLOAT_EQ(-9.7928059e-06f, combine_weights.difference(18, 6));
  EXPECT_FLOAT_EQ(8.7106271e-21f, combine_weights.difference(19, 6));
  EXPECT_FLOAT_EQ(-0.024550479f, combine_weights.difference(0, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(1, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 7));
  EXPECT_FLOAT_EQ(0.0089381579f, combine_weights.difference(3, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 7));
  EXPECT_FLOAT_EQ(7.35411e-20f, combine_weights.difference(6, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 7));
  EXPECT_FLOAT_EQ(-0.027424494f, combine_weights.difference(8, 7));
  EXPECT_FLOAT_EQ(1.0885018e-19f, combine_weights.difference(9, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 7));
  EXPECT_FLOAT_EQ(5.2057445e-21f, combine_weights.difference(11, 7));
  EXPECT_FLOAT_EQ(-0.060075536f, combine_weights.difference(12, 7));
  EXPECT_FLOAT_EQ(0.011130749f, combine_weights.difference(13, 7));
  EXPECT_FLOAT_EQ(-0.023096029f, combine_weights.difference(14, 7));
  EXPECT_FLOAT_EQ(-0.0099584702f, combine_weights.difference(15, 7));
  EXPECT_FLOAT_EQ(0.0023827187f, combine_weights.difference(16, 7));
  EXPECT_FLOAT_EQ(7.0092086e-21f, combine_weights.difference(17, 7));
  EXPECT_FLOAT_EQ(0.013480823f, combine_weights.difference(18, 7));
  EXPECT_FLOAT_EQ(0.012368691f, combine_weights.difference(19, 7));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(0, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(1, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 8));
  EXPECT_FLOAT_EQ(-1.2916376e-19f, combine_weights.difference(3, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 8));
  EXPECT_FLOAT_EQ(-7.0785794e-20f, combine_weights.difference(6, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(7, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(8, 8));
  EXPECT_FLOAT_EQ(-1.0477198e-19f, combine_weights.difference(9, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 8));
  EXPECT_FLOAT_EQ(-5.0107047e-21f, combine_weights.difference(11, 8));
  EXPECT_FLOAT_EQ(5.8620719e-21f, combine_weights.difference(12, 8));
  EXPECT_FLOAT_EQ(-8.1450391e-21f, combine_weights.difference(13, 8));
  EXPECT_FLOAT_EQ(2.8448266e-21f, combine_weights.difference(14, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(15, 8));
  EXPECT_FLOAT_EQ(-7.380124e-21f, combine_weights.difference(16, 8));
  EXPECT_FLOAT_EQ(-6.7465994e-21f, combine_weights.difference(17, 8));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(18, 8));
  EXPECT_FLOAT_EQ(5.1251339e-21f, combine_weights.difference(19, 8));
  EXPECT_FLOAT_EQ(-0.012235705f, combine_weights.difference(0, 9));
  EXPECT_FLOAT_EQ(1.7338985e-05f, combine_weights.difference(1, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(2, 9));
  EXPECT_FLOAT_EQ(0.0044899601f, combine_weights.difference(3, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(4, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(5, 9));
  EXPECT_FLOAT_EQ(3.8555325e-20f, combine_weights.difference(6, 9));
  EXPECT_FLOAT_EQ(2.031264e-05f, combine_weights.difference(7, 9));
  EXPECT_FLOAT_EQ(-0.013668084f, combine_weights.difference(8, 9));
  EXPECT_FLOAT_EQ(2.8104352e-05f, combine_weights.difference(9, 9));
  EXPECT_FLOAT_EQ(0.0f, combine_weights.difference(10, 9));
  EXPECT_FLOAT_EQ(1.7328591e-06f, combine_weights.difference(11, 9));
  EXPECT_FLOAT_EQ(-0.029941026f, combine_weights.difference(12, 9));
  EXPECT_FLOAT_EQ(0.0055463808f, combine_weights.difference(13, 9));
  EXPECT_FLOAT_EQ(-0.011510822f, combine_weights.difference(14, 9));
  EXPECT_FLOAT_EQ(-0.0049568322f, combine_weights.difference(15, 9));
  EXPECT_FLOAT_EQ(0.0011868662f, combine_weights.difference(16, 9));
  EXPECT_FLOAT_EQ(3.6747111e-21f, combine_weights.difference(17, 9));
  EXPECT_FLOAT_EQ(0.0067189396f, combine_weights.difference(18, 9));
  EXPECT_FLOAT_EQ(0.0061644278f, combine_weights.difference(19, 9));

  EXPECT_FLOAT_EQ(0.0f, combine_bias.difference(0));
  EXPECT_FLOAT_EQ(-0.002623301f, combine_bias.difference(1));
  EXPECT_FLOAT_EQ(-2.227431e-21f, combine_bias.difference(2));
  EXPECT_FLOAT_EQ(-0.007554865f, combine_bias.difference(3));
  EXPECT_FLOAT_EQ(0.0f, combine_bias.difference(4));
  EXPECT_FLOAT_EQ(-9.1689223e-21f, combine_bias.difference(5));
  EXPECT_FLOAT_EQ(-0.00025963833f, combine_bias.difference(6));
  EXPECT_FLOAT_EQ(-0.031239605f, combine_bias.difference(7));
  EXPECT_FLOAT_EQ(-6.002357e-21f, combine_bias.difference(8));
  EXPECT_FLOAT_EQ(-0.015563218f, combine_bias.difference(9));
}
