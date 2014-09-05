#ifndef AUTOENCODER_RECURRENTNEURALNETWORKPARTOFSPEECHTAGGER_HPP_
#define AUTOENCODER_RECURRENTNEURALNETWORKPARTOFSPEECHTAGGER_HPP_

#include <random>
#include <vector>

#include "blob.hpp"
#include "euclideanlosslayer.hpp"
#include "partofspeechtagger.hpp"
#include "partofspeechsentencelayer.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  class LookupTable;

  class RecurrentNeuralNetworkPartOfSpeechTagger : public PartOfSpeechTagger {
  public:
    RecurrentNeuralNetworkPartOfSpeechTagger(
      LookupTable &word_table, LookupTable &tag_table, float p, std::mt19937 &generator);

    virtual ~RecurrentNeuralNetworkPartOfSpeechTagger() = default;

    void ForwardBackwardCpu(const TaggedSentence &tagged_sentence);

    void Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        float learning_rate,
        int iterations) override;

    void Validate(const std::vector<TaggedSentence> &tagged_sentences) const override;

    std::vector<std::string> Tag(const std::vector<std::string> &sentence) const override;

    float ScoreTagging(const TaggedSentence &tagged_sentence) const override;

  public:
    LookupTable &word_table;
    LookupTable &tag_table;
    float p;
    std::mt19937 &generator;
    Blob classify_weights, classify_bias;
    Blob combine_weights, combine_bias;
    PartOfSpeechSentenceLayer part_of_speech_sentence;
    EuclideanLossLayer loss;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECURRENTNEURALNETWORKPARTOFSPEECHTAGGER_HPP_
