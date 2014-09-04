#ifndef AUTOENCODER_RECURRENTNEURALNETWORKPARTOFSPEECHTAGGER_HPP_
#define AUTOENCODER_RECURRENTNEURALNETWORKPARTOFSPEECHTAGGER_HPP_

#include "partofspeechtagger.hpp"

namespace autoencoder {

  class RecurrentNeuralNetworkPartOfSpeechTagger : public PartOfSpeechTagger {
  public:
    RecurrentNeuralNetworkPartOfSpeechTagger() = default;

    virtual ~RecurrentNeuralNetworkPartOfSpeechTagger() = default;

    void Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        float learning_rate,
        int iterations) override;

    void Validate(const std::vector<TaggedSentence> &tagged_sentences) const override;

    std::vector<std::string> Tag(const std::vector<std::string> &sentence) const override;

    float ScoreTagging(const TaggedSentence &tagged_sentence) const override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECURRENTNEURALNETWORKPARTOFSPEECHTAGGER_HPP_
