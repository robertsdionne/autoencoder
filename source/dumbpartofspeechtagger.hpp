#ifndef AUTOENCODER_DUMBPARTOFSPEECHTAGGER_HPP_
#define AUTOENCODER_DUMBPARTOFSPEECHTAGGER_HPP_

#include "partofspeechtagger.hpp"

namespace autoencoder {

  class DumbPartOfSpeechTagger : public PartOfSpeechTagger {
  public:
    DumbPartOfSpeechTagger() = default;

    virtual ~DumbPartOfSpeechTagger() = default;

    void Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        float learning_rate,
        int iterations) override;

    void Validate(const std::vector<TaggedSentence> &tagged_sentences) const override;

    std::vector<std::string> Tag(const std::vector<std::string> &sentence) const override;

    float ScoreTagging(const TaggedSentence &tagged_sentence) const override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_DUMBPARTOFSPEECHTAGGER_HPP_
