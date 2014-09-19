#ifndef AUTOENCODER_PARTOFSPEECHTAGGER_HPP_
#define AUTOENCODER_PARTOFSPEECHTAGGER_HPP_

#include <string>
#include <vector>
#include <unordered_set>

#include "interface.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  template <typename F>
  class Evaluator;

  template <typename F>
  class PartOfSpeechTagger {
    DECLARE_INTERFACE(PartOfSpeechTagger);

  public:
    virtual void Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        F learning_rate,
        F momentum,
        int iterations,
        Evaluator<F> &evaluator,
        const std::vector<TaggedSentence> &validation_sentences,
        const std::unordered_set<std::string> &training_vocabulary) = 0;

    virtual void Validate(const std::vector<TaggedSentence> &tagged_sentences) const = 0;

    virtual std::vector<std::string> Tag(const std::vector<std::string> &sentence) = 0;

    virtual F ScoreTagging(const TaggedSentence &tagged_sentence) const = 0;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHTAGGER_HPP_
