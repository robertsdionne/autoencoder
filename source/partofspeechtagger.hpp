#ifndef AUTOENCODER_PARTOFSPEECHTAGGER_H_
#define AUTOENCODER_PARTOFSPEECHTAGGER_H_

#include <string>
#include <vector>

#include "interface.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  class PartOfSpeechTagger {
    DECLARE_INTERFACE(PartOfSpeechTagger);

  public:
    virtual void Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        float learning_rate,
        int iterations) = 0;

    virtual void Validate(const std::vector<TaggedSentence> &tagged_sentences) const = 0;

    virtual std::vector<std::string> Tag(const std::vector<std::string> &sentence) const = 0;

    virtual float ScoreTagging(const TaggedSentence &tagged_sentence) const = 0;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHTAGGER_H_
