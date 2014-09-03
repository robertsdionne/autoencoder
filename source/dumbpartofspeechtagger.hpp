#ifndef AUTOENCODER_DUMBPARTOFSPEECHTAGGER_HPP_
#define AUTOENCODER_DUMBPARTOFSPEECHTAGGER_HPP_

#include "partofspeechtagger.hpp"

namespace autoencoder {

  class DumbPartOfSpeechTagger : public PartOfSpeechTagger {
  public:
    DumbPartOfSpeechTagger() = default;

    virtual ~DumbPartOfSpeechTagger() = default;

    virtual void Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        float learning_rate,
        int iterations);

    virtual void Validate(const std::vector<TaggedSentence> &tagged_sentences) const;

    virtual std::vector<std::string> Tag(const std::vector<std::string> &sentence) const;

    virtual float ScoreTagging(const TaggedSentence &tagged_sentence) const;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_DUMBPARTOFSPEECHTAGGER_HPP_
