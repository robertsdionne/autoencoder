#include <cassert>
#include <cmath>
#include <random>

#include "blob.hpp"
#include "partofspeechwordlayer.hpp"

namespace autoencoder {

  PartOfSpeechWordLayer::PartOfSpeechWordLayer(
      float p,
      Blob &classify_weights, Blob &classify_bias,
      Blob &combine_weights, Blob &combine_bias,
      std::mt19937 &generator)
    : dropout(p, generator),
      corrupted_recurrent(combine_weights.height),
      corrupted_word(combine_weights.width - combine_weights.height),
      classify(classify_weights, classify_bias), classified(classify_weights.height),
      softmax(),
      concatenate(), concatenated(combine_weights.width),
      combine(combine_weights, combine_bias), combined(combine_weights.height),
      rectified_linear() {}

  float PartOfSpeechWordLayer::ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) {
    // std::cout << "corrupted_recurrent.width " << corrupted_recurrent.width << std::endl;
    // std::cout << "corrupted_word.width " << corrupted_word.width << std::endl;
    auto dropout_output = Blobs{&corrupted_recurrent, &corrupted_word};
    auto dropout_recurrent_output = Blobs{&corrupted_recurrent};
    auto classify_output = Blobs{&classified};
    auto softmax_output = Blobs{top->at(0)};
    auto concatenate_output = Blobs{&concatenated};
    auto combine_output = Blobs{&combined};
    auto rectified_linear_output = Blobs{top->at(1)};

    dropout.ForwardCpu(mode, bottom, &dropout_output);
    classify.ForwardCpu(mode, dropout_recurrent_output, &classify_output);
    softmax.ForwardCpu(mode, classify_output, &softmax_output);
    concatenate.ForwardCpu(mode, dropout_output, &concatenate_output);
    combine.ForwardCpu(mode, concatenate_output, &combine_output);
    rectified_linear.ForwardCpu(mode, combine_output, &rectified_linear_output);
    top->at(0)->IsValid();
    top->at(1)->IsValid();
    return 0.0f;
  }

  void PartOfSpeechWordLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    auto dropout_output = Blobs{&corrupted_recurrent, &corrupted_word};
    auto dropout_recurrent_output = Blobs{&corrupted_recurrent};
    auto classify_output = Blobs{&classified};
    auto softmax_output = Blobs{top.at(0)};
    auto concatenate_output = Blobs{&concatenated};
    auto combine_output = Blobs{&combined};
    auto rectified_linear_output = Blobs{top.at(1)};

    rectified_linear.BackwardCpu(rectified_linear_output, &combine_output);
    combine.BackwardCpu(combine_output, &concatenate_output);
    concatenate.BackwardCpu(concatenate_output, &dropout_output);
    softmax.BackwardCpu(softmax_output, &classify_output);
    classify.BackwardCpu(classify_output, &dropout_recurrent_output);
    dropout.BackwardCpu(dropout_output, bottom);
    bottom->at(0)->IsValid();
    bottom->at(1)->IsValid();
  }

}  // namespace autoencoder
