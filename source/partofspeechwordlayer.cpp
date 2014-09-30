#include <cassert>
#include <cmath>
#include <random>

#include "blob.hpp"
#include "device.hpp"
#include "partofspeechwordlayer.hpp"

namespace autoencoder {

  template <typename F>
  PartOfSpeechWordLayer<F>::PartOfSpeechWordLayer(
      Device<F> &device,
      F p,
      Blob<F> &classify_weights, Blob<F> &classify_bias,
      Blob<F> &combine_weights, Blob<F> &combine_bias,
      std::mt19937 &generator)
    : dropout(p, generator),
      corrupted_recurrent(combine_weights.height),
      corrupted_word(combine_weights.width - combine_weights.height),
      classify(device, classify_weights, classify_bias), classified(classify_weights.height),
      softmax(device),
      concatenate(), concatenated(combine_weights.width),
      combine(device, combine_weights, combine_bias), combined(combine_weights.height),
      rectified_linear(device) {}

  template <typename F>
  F PartOfSpeechWordLayer<F>::ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    // std::cout << "corrupted_recurrent.width " << corrupted_recurrent.width << std::endl;
    // std::cout << "corrupted_word.width " << corrupted_word.width << std::endl;
    auto dropout_output = Blobs<F>{&corrupted_recurrent, &corrupted_word};
    auto dropout_recurrent_output = Blobs<F>{&corrupted_recurrent};
    auto classify_output = Blobs<F>{&classified};
    auto softmax_output = Blobs<F>{top->at(0)};
    auto concatenate_output = Blobs<F>{&concatenated};
    auto combine_output = Blobs<F>{&combined};
    auto rectified_linear_output = Blobs<F>{top->at(1)};

    dropout.ForwardCpu(mode, bottom, &dropout_output);
    classify.ForwardXpu(mode, dropout_recurrent_output, &classify_output);
    softmax.ForwardXpu(mode, classify_output, &softmax_output);
    concatenate.ForwardCpu(mode, dropout_output, &concatenate_output);
    combine.ForwardXpu(mode, concatenate_output, &combine_output);
    rectified_linear.ForwardXpu(mode, combine_output, &rectified_linear_output);
    top->at(0)->IsValid();
    top->at(1)->IsValid();
    return 0.0f;
  }

  template <typename F>
  void PartOfSpeechWordLayer<F>::BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {
    auto dropout_output = Blobs<F>{&corrupted_recurrent, &corrupted_word};
    auto dropout_recurrent_output = Blobs<F>{&corrupted_recurrent};
    auto classify_output = Blobs<F>{&classified};
    auto softmax_output = Blobs<F>{top.at(0)};
    auto concatenate_output = Blobs<F>{&concatenated};
    auto combine_output = Blobs<F>{&combined};
    auto rectified_linear_output = Blobs<F>{top.at(1)};

    rectified_linear.BackwardXpu(rectified_linear_output, &combine_output);
    combine.BackwardXpu(combine_output, &concatenate_output);
    concatenate.BackwardCpu(concatenate_output, &dropout_output);
    softmax.BackwardXpu(softmax_output, &classify_output);
    classify.BackwardXpu(classify_output, &dropout_recurrent_output);
    dropout.BackwardCpu(dropout_output, bottom);
    bottom->at(0)->IsValid();
    bottom->at(1)->IsValid();
  }

  template class PartOfSpeechWordLayer<float>;
  template class PartOfSpeechWordLayer<double>;

}  // namespace autoencoder
