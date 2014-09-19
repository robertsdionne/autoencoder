#ifndef AUTOENCODER_DUMBPARTOFSPEECHTAGGER_HPP_
#define AUTOENCODER_DUMBPARTOFSPEECHTAGGER_HPP_

#include <vector>
#include <unordered_set>

#include "partofspeechtagger.hpp"

namespace autoencoder {

  template <typename F>
  class Evaluator;

  template <typename F>
  class DumbPartOfSpeechTagger : public PartOfSpeechTagger<F> {
  public:
    DumbPartOfSpeechTagger() = default;

    virtual ~DumbPartOfSpeechTagger() = default;

    void Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        F learning_rate,
        F momentum,
        int iterations,
        Evaluator<F> &evaluator,
        const std::vector<TaggedSentence> &validation_sentences,
        const std::unordered_set<std::string> &training_vocabulary) override;

    void Validate(const std::vector<TaggedSentence> &tagged_sentences) const override;

    std::vector<std::string> Tag(const std::vector<std::string> &sentence) override;

    F ScoreTagging(const TaggedSentence &tagged_sentence) const override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_DUMBPARTOFSPEECHTAGGER_HPP_
