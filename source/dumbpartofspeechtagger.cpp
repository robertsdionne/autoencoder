#include <limits>
#include <string>
#include <vector>
#include <unordered_set>

#include "dumbpartofspeechtagger.hpp"
#include "evaluator.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  template <typename F>
  void DumbPartOfSpeechTagger<F>::Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        F learning_rate,
        F momentum,
        int iterations,
        Evaluator<F> &evaluator,
        const std::vector<TaggedSentence> &validation_sentences,
        const std::unordered_set<std::string> &training_vocabulary) {
  }

  template <typename F>
  void DumbPartOfSpeechTagger<F>::Validate(const std::vector<TaggedSentence> &tagged_sentences) const {
  }

  template <typename F>
  std::vector<std::string> DumbPartOfSpeechTagger<F>::Tag(
      const std::vector<std::string> &sentence) {
    auto tags = std::vector<std::string>();
    for (auto &word : sentence) {
      tags.push_back("NN");
    }
    return tags;
  }

  template <typename F>
  F DumbPartOfSpeechTagger<F>::ScoreTagging(const TaggedSentence &tagged_sentence) const {
    return -std::numeric_limits<F>::infinity();
  }

  template class DumbPartOfSpeechTagger<float>;
  template class DumbPartOfSpeechTagger<double>;

}  // namespace autoencoder
