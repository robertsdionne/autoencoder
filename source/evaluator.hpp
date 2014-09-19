#ifndef AUTOENCODER_EVALUATOR_HPP_
#define AUTOENCODER_EVALUATOR_HPP_

#include <string>
#include <unordered_set>
#include <vector>

#include "taggedsentence.hpp"

namespace autoencoder {

  template <typename F>
  class PartOfSpeechTagger;

  template <typename F>
  struct EvaluationReport {
    F tag_accuracy, unknown_accuracy;
  };

  template <typename F>
  class Evaluator {
  public:
    Evaluator() = default;

    virtual ~Evaluator() = default;

    EvaluationReport<F> Evaluate(
        PartOfSpeechTagger<F> &part_of_speech_tagger,
        const std::vector<TaggedSentence> &tagged_sentences,
        const std::unordered_set<std::string> &training_vocabulary) const;
  };

  template <typename F>
  std::ostream &operator <<(std::ostream &out, const EvaluationReport<F> &report);

}  // namespace autoencoder

#endif  // AUTOENCODER_EVALUATOR_HPP_
