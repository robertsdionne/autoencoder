#include <limits>
#include <random>
#include <vector>

#include "blob.hpp"
#include "lookuptable.hpp"
#include "recurrentneuralnetworkpartofspeechtagger.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {
  RecurrentNeuralNetworkPartOfSpeechTagger::RecurrentNeuralNetworkPartOfSpeechTagger(
      LookupTable &word_table, LookupTable &tag_table, float p, std::mt19937 &generator)
  : word_table(word_table), tag_table(tag_table),
    p(p), generator(generator),
    classify_weights(), classify_bias(),
    combine_weights(), combine_bias(),
    part_of_speech_sentence(
        p, classify_weights, classify_bias, combine_weights, combine_bias, generator),
    loss() {}

  void RecurrentNeuralNetworkPartOfSpeechTagger::ForwardBackwardCpu(
      const TaggedSentence &tagged_sentence) {
    auto word_vectors = autoencoder::Blobs{};
    word_table.ForwardCpu(tagged_sentence.words, &word_vectors);

  }

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
