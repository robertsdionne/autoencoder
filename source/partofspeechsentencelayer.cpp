#include <cmath>
#include <random>

#include "blob.hpp"
#include "partofspeechsentencelayer.hpp"

namespace autoencoder {

  PartOfSpeechSentenceLayer::PartOfSpeechSentenceLayer(
      float p,
      Blob &classify_weights, Blob &classify_bias,
      Blob &combine_weights, Blob &combine_bias,
      std::mt19937 &generator)
    : p(p), generator(generator),
      classify_weights(classify_weights), classify_bias(classify_bias),
      combine_weights(combine_weights), combine_bias(combine_bias) {}

  void PartOfSpeechSentenceLayer::ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) {
    layers.clear();
    recurrent_states.clear();

    recurrent_states.emplace_back(combine_weights.height);

    for (auto i = 0; i < combine_weights.height; ++i) {
      recurrent_states.back().value(i) = bottom.at(0)->value(i);
    }

    for (auto i = 1; i < bottom.size(); ++i) {
      layers.emplace_back(
          p, classify_weights, classify_bias, combine_weights, combine_bias, generator);
      recurrent_states.emplace_back(combine_weights.height);

      auto layer_input = Blobs{&recurrent_states.at(i - 1), bottom.at(i)};
      auto layer_output = Blobs{top->at(i - 1), &recurrent_states.at(i)};

      // std::cout << "layers.at(" << i << " - 1).ForwardCpu" << std::endl;
      layers.at(i - 1).ForwardCpu(mode, layer_input, &layer_output);
      top->at(i - 1)->IsValid();
    }

    for (auto i = 0; i < combine_weights.height; ++i) {
      top->back()->value(i) = recurrent_states.back().value(i);
    }
    top->back()->IsValid();
  }

  void PartOfSpeechSentenceLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = bottom->size() - 1; i > 0; --i) {
      auto layer_input = Blobs{&recurrent_states.at(i - 1), bottom->at(i)};
      auto layer_output = Blobs{top.at(i - 1), &recurrent_states.at(i)};

      // std::cout << "layers.at(" << i << " - 1).BackwardCpu" << std::endl;
      layers.at(i - 1).BackwardCpu(layer_output, &layer_input);
      bottom->at(i)->IsValid();
    }

    for (auto i = 0; i < combine_weights.height; ++i) {
      bottom->front()->difference(i) = recurrent_states.front().difference(i);
    }
    bottom->front()->IsValid();
  }

}  // namespace autoencoder
