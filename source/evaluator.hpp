#ifndef AUTOENCODER_EVALUATOR_HPP_
#define AUTOENCODER_EVALUATOR_HPP_

#include <string>
#include <unordered_set>
#include <vector>

#include "taggedsentence.hpp"

namespace autoencoder {

  class PartOfSpeechTagger;

  struct EvaluationReport {
    float tag_accuracy, unknown_accuracy;
  };

  class Evaluator {
  public:
    Evaluator() = default;

    virtual ~Evaluator() = default;

    EvaluationReport Evaluate(
        PartOfSpeechTagger &part_of_speech_tagger,
        const std::vector<TaggedSentence> &tagged_sentences,
        const std::unordered_set<std::string> &training_vocabulary) const;
  };

  std::ostream &operator <<(std::ostream &out, const EvaluationReport &report);

}  // namespace autoencoder

#endif  // AUTOENCODER_EVALUATOR_HPP_
