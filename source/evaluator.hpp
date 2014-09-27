#ifndef AUTOENCODER_EVALUATOR_HPP_
#define AUTOENCODER_EVALUATOR_HPP_

#include <random>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "taggedsentence.hpp"

namespace autoencoder {

  template <typename F>
  class PartOfSpeechTagger;

  template <typename F>
  struct EvaluationReport {
    F tag_accuracy, tag_accuracy_delta, unknown_accuracy, unknown_accuracy_delta;
    std::unordered_map<int, F> histogram;
    std::unordered_map<int, F> length_histogram;
  };

  template <typename F>
  class Evaluator {
  public:
    Evaluator(std::mt19937 &generator);

    virtual ~Evaluator() = default;

    EvaluationReport<F> Evaluate(
        PartOfSpeechTagger<F> &part_of_speech_tagger,
        const std::vector<TaggedSentence> &tagged_sentences,
        const std::unordered_set<std::string> &training_vocabulary);

  private:
    F previous_tag_accuracy, previous_unknown_accuracy;
    std::mt19937 &generator;
  };

  template <typename F>
  std::ostream &operator <<(std::ostream &out, EvaluationReport<F> &report);

}  // namespace autoencoder

#endif  // AUTOENCODER_EVALUATOR_HPP_
