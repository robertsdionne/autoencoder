#include <cmath>

#include "blob.hpp"
#include "partofspeechsentencelayer.hpp"

namespace autoencoder {

  PartOfSpeechSentenceLayer::PartOfSpeechSentenceLayer(
      float p,
      Blob &classify_weights, Blob &classify_bias,
      Blob &combine_weights, Blob &combine_bias)
    : p(p),
      classify_weights(classify_weights), classify_bias(classify_bias),
      combine_weights(combine_weights), combine_bias(combine_bias) {}

  void PartOfSpeechSentenceLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
  }

  void PartOfSpeechSentenceLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
  }

}  // namespace autoencoder
