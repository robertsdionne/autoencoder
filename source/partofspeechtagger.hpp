#ifndef AUTOENCODER_PARTOFSPEECHTAGGER_HPP_
#define AUTOENCODER_PARTOFSPEECHTAGGER_HPP_

#include <string>
#include <vector>
#include <unordered_set>

#include "interface.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  class Evaluator;

  class PartOfSpeechTagger {
    DECLARE_INTERFACE(PartOfSpeechTagger);

  public:
    virtual void Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        float learning_rate,
        float momentum,
        int iterations,
        Evaluator &evaluator,
        const std::vector<TaggedSentence> &validation_sentences,
        const std::unordered_set<std::string> &training_vocabulary) = 0;

    virtual void Validate(const std::vector<TaggedSentence> &tagged_sentences) const = 0;

    virtual std::vector<std::string> Tag(const std::vector<std::string> &sentence) = 0;

    virtual float ScoreTagging(const TaggedSentence &tagged_sentence) const = 0;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHTAGGER_HPP_
