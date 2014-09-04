#include "recurrentneuralnetworkpartofspeechtagger.hpp"

namespace autoencoder {

  void RecurrentNeuralNetworkPartOfSpeechTagger::Train(
      const std::vector<TaggedSentence> &tagged_sentences,
      float learning_rate,
      int iterations) {
  }

  void RecurrentNeuralNetworkPartOfSpeechTagger::Validate(
      const std::vector<TaggedSentence> &tagged_sentences) const {
  }

  std::vector<std::string> RecurrentNeuralNetworkPartOfSpeechTagger::Tag(
      const std::vector<std::string> &sentence) const {
    auto tags = std::vector<std::string>();
    for (auto &word : sentence) {
      tags.push_back("NN");
    }
    return tags;
  }

  float RecurrentNeuralNetworkPartOfSpeechTagger::ScoreTagging(
      const TaggedSentence &tagged_sentence) const {
    return -std::numeric_limits<float>::infinity();
  }

}  // namespace autoencoder
