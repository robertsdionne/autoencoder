#include <cmath>
#include <random>

#include "blob.hpp"
#include "device.hpp"
#include "partofspeechsentencelayer.hpp"

namespace autoencoder {

  template <typename F>
  PartOfSpeechSentenceLayer<F>::PartOfSpeechSentenceLayer(
      Device<F> &device,
      F p,
      Blob<F> &classify_weights, Blob<F> &classify_bias,
      Blob<F> &combine_weights, Blob<F> &combine_bias,
      std::mt19937 &generator)
    : device(device), p(p), generator(generator),
      classify_weights(classify_weights), classify_bias(classify_bias),
      combine_weights(combine_weights), combine_bias(combine_bias) {}

  template <typename F>
  F PartOfSpeechSentenceLayer<F>::ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    layers.clear();
    recurrent_states.clear();

    recurrent_states.emplace_back(combine_weights.height);

    device.Initialize(recurrent_states.back());
    device.Ship(recurrent_states.back());

    device.Copy(bottom.at(0)->values, &recurrent_states.back().values);

    for (auto i = 1; i < bottom.size(); ++i) {
      layers.emplace_back(
          device, p, classify_weights, classify_bias, combine_weights, combine_bias, generator);
      recurrent_states.emplace_back(combine_weights.height);

      device.Initialize(recurrent_states.back());
      device.Ship(recurrent_states.back());

      auto layer_input = Blobs<F>{&recurrent_states.at(i - 1), bottom.at(i)};
      auto layer_output = Blobs<F>{top->at(i - 1), &recurrent_states.at(i)};

      // std::cout << "layers.at(" << i << " - 1).ForwardCpu" << std::endl;
      layers.at(i - 1).ForwardCpu(mode, layer_input, &layer_output);
      // top->at(i - 1)->IsValid();
    }

    device.Copy(recurrent_states.back().values, &top->back()->values);
    // top->back()->IsValid();
    return 0.0f;
  }

  template <typename F>
  void PartOfSpeechSentenceLayer<F>::BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {
    for (auto i = bottom->size() - 1; i > 0; --i) {
      auto layer_input = Blobs<F>{&recurrent_states.at(i - 1), bottom->at(i)};
      auto layer_output = Blobs<F>{top.at(i - 1), &recurrent_states.at(i)};

      // std::cout << "layers.at(" << i << " - 1).BackwardCpu" << std::endl;
      layers.at(i - 1).BackwardCpu(layer_output, &layer_input);
      // bottom->at(i)->IsValid();
    }

    device.Copy(recurrent_states.front().differences, &bottom->front()->differences);
    // bottom->front()->IsValid();
  }

  template class PartOfSpeechSentenceLayer<float>;
  template class PartOfSpeechSentenceLayer<double>;

}  // namespace autoencoder
