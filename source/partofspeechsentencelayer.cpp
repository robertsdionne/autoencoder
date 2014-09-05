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

  void PartOfSpeechSentenceLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    layers.clear();
    recurrent_states.clear();

    recurrent_states.emplace_back(bottom.at(0)->width);

    for (auto i = 0; i < bottom.at(0)->width; ++i) {
      recurrent_states.back().value(i) = bottom.at(0)->value(i);
    }

    for (auto i = 1; i < bottom.size(); ++i) {
      layers.emplace_back(
          p, classify_weights, classify_bias, combine_weights, combine_bias, generator);
      recurrent_states.emplace_back(bottom.at(0)->width);

      auto layer_input = Blobs{&recurrent_states.at(i - 1), bottom.at(i)};
      auto layer_output = Blobs{top->at(i - 1), &recurrent_states.at(i)};

      layers.at(i - 1).ForwardCpu(layer_input, &layer_output);
    }

    for (auto i = 0; i < recurrent_states.back().width; ++i) {
      top->back()->value(i) = recurrent_states.back().value(i);
    }
  }

  void PartOfSpeechSentenceLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = bottom->size() - 1; i > 0; --i) {
      auto layer_input = Blobs{&recurrent_states.at(i - 1), bottom->at(i)};
      auto layer_output = Blobs{top.at(i - 1), &recurrent_states.at(i)};

      layers.at(i - 1).BackwardCpu(layer_output, &layer_input);
    }

    for (auto i = 0; i < recurrent_states.front().width; ++i) {
      bottom->front()->difference(i) = recurrent_states.front().difference(i);
    }
  }

}  // namespace autoencoder
