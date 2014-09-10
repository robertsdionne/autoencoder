#include <limits>
#include <string>
#include <vector>
#include <unordered_set>

#include "dumbpartofspeechtagger.hpp"
#include "evaluator.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  void DumbPartOfSpeechTagger::Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        float learning_rate,
        float momentum,
        int iterations,
        Evaluator &evaluator,
        const std::vector<TaggedSentence> &validation_sentences,
        const std::unordered_set<std::string> &training_vocabulary) {
  }

  void DumbPartOfSpeechTagger::Validate(const std::vector<TaggedSentence> &tagged_sentences) const {
  }

  std::vector<std::string> DumbPartOfSpeechTagger::Tag(
      const std::vector<std::string> &sentence) {
    auto tags = std::vector<std::string>();
    for (auto &word : sentence) {
      tags.push_back("NN");
    }
    return tags;
  }

  float DumbPartOfSpeechTagger::ScoreTagging(const TaggedSentence &tagged_sentence) const {
    return -std::numeric_limits<float>::infinity();
  }

}  // namespace autoencoder
