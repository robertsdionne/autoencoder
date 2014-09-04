#include <cmath>

#include "blob.hpp"
#include "partofspeechtaglayer.hpp"

namespace autoencoder {

  PartOfSpeechTagLayer::PartOfSpeechTagLayer(
      float p,
      Blob &classify_weights, Blob &classify_bias,
      Blob &combine_weights, Blob &combine_bias)
    : concatenate(), dropout(p),
      classify(classify_weights, classify_bias),
      combine(combine_weights, combine_bias),
      rectified_linear(), softmax() {}

  void PartOfSpeechTagLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    concatenated.Reshape(bottom.at(0)->width + bottom.at(1)->width);
    corrupted.Reshape(concatenated.width);
    classified.Reshape(top->at(0)->width);
    combined.Reshape(bottom.at(0)->width);

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

  void PartOfSpeechTagLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
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
