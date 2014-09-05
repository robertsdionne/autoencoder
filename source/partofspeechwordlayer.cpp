#include <cmath>

#include "blob.hpp"
#include "partofspeechwordlayer.hpp"

namespace autoencoder {

  PartOfSpeechWordLayer::PartOfSpeechWordLayer(
      float p,
      Blob &classify_weights, Blob &classify_bias,
      Blob &combine_weights, Blob &combine_bias,
      unsigned int random_seed)
    : dropout(p, random_seed),
      corrupted_recurrent(combine_weights.height),
      corrupted_word(combine_weights.width - combine_weights.height),
      classify(classify_weights, classify_bias), classified(classify_weights.height),
      softmax(),
      concatenate(), concatenated(combine_weights.width),
      combine(combine_weights, combine_bias), combined(combine_weights.height),
      rectified_linear() {}

  void PartOfSpeechWordLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    auto dropout_output = Blobs{&corrupted_recurrent, &corrupted_word};
    auto dropout_recurrent_output = Blobs{&corrupted_recurrent};
    auto classify_output = Blobs{&classified};
    auto softmax_output = Blobs{top->at(0)};
    auto concatenate_output = Blobs{&concatenated};
    auto combine_output = Blobs{&combined};
    auto rectified_linear_output = Blobs{top->at(1)};

    dropout.ForwardCpu(bottom, &dropout_output);
    classify.ForwardCpu(dropout_recurrent_output, &classify_output);
    softmax.ForwardCpu(classify_output, &softmax_output);
    concatenate.ForwardCpu(dropout_output, &concatenate_output);
    combine.ForwardCpu(concatenate_output, &combine_output);
    rectified_linear.ForwardCpu(combine_output, &rectified_linear_output);
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
  }

}  // namespace autoencoder
