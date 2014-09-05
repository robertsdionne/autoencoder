#include <cmath>

#include "blob.hpp"
#include "partofspeechwordlayer.hpp"

namespace autoencoder {

  PartOfSpeechWordLayer::PartOfSpeechWordLayer(
      float p,
      Blob &classify_weights, Blob &classify_bias,
      Blob &combine_weights, Blob &combine_bias,
      unsigned int random_seed)
    : concatenate(), concatenated(combine_weights.width),
      dropout(p, random_seed), corrupted(combine_weights.width),
      classify(classify_weights, classify_bias), classified(classify_weights.height),
      combine(combine_weights, combine_bias), combined(combine_weights.height),
      rectified_linear(), softmax() {}

  void PartOfSpeechWordLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    auto concatenate_output = Blobs{&concatenated};
    auto dropout_output = Blobs{&corrupted};
    auto classify_output = Blobs{&classified};
    auto softmax_output = Blobs{top->at(0)};
    auto combine_output = Blobs{&combined};
    auto rectified_linear_output = Blobs{top->at(1)};

    concatenate.ForwardCpu(bottom, &concatenate_output);
    dropout.ForwardCpu(concatenate_output, &dropout_output);
    classify.ForwardCpu(dropout_output, &classify_output);
    softmax.ForwardCpu(classify_output, &softmax_output);
    combine.ForwardCpu(dropout_output, &combine_output);
    rectified_linear.ForwardCpu(combine_output, &rectified_linear_output);
  }

  void PartOfSpeechWordLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    auto concatenate_output = Blobs{&concatenated};
    auto dropout_output = Blobs{&corrupted};
    auto classify_output = Blobs{&classified};
    auto softmax_output = Blobs{top.at(0)};
    auto combine_output = Blobs{&combined};
    auto rectified_linear_output = Blobs{top.at(1)};

    rectified_linear.BackwardCpu(rectified_linear_output, &combine_output);
    combine.BackwardCpu(combine_output, &dropout_output);
    softmax.BackwardCpu(softmax_output, &classify_output);
    classify.BackwardCpu(classify_output, &dropout_output);
    dropout.BackwardCpu(dropout_output, &concatenate_output);
    concatenate.BackwardCpu(concatenate_output, bottom);
  }

}  // namespace autoencoder
