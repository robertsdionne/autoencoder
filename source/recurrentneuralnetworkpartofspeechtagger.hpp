#ifndef AUTOENCODER_RECURRENTNEURALNETWORKPARTOFSPEECHTAGGER_HPP_
#define AUTOENCODER_RECURRENTNEURALNETWORKPARTOFSPEECHTAGGER_HPP_

#include <random>
#include <vector>
#include <unordered_set>

#include "blob.hpp"
#include "euclideanlosslayer.hpp"
#include "partofspeechtagger.hpp"
#include "partofspeechsentencelayer.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  class Evaluator;
  class LookupTable;

  class RecurrentNeuralNetworkPartOfSpeechTagger : public PartOfSpeechTagger {
  public:
    RecurrentNeuralNetworkPartOfSpeechTagger(
      LookupTable &word_table, LookupTable &tag_table,
      float p, std::mt19937 &generator,
      int recurrent_state_dimension,
      int tag_dimension,
      int word_representation_dimension);

    virtual ~RecurrentNeuralNetworkPartOfSpeechTagger() = default;

    void ForwardCpu(const std::vector<std::string> &sentence, std::vector<std::string> *tags);

    void ForwardBackwardCpu(const TaggedSentence &tagged_sentence);

    void Train(
        const std::vector<TaggedSentence> &tagged_sentences,
        float learning_rate,
        int iterations,
        Evaluator &evaluator,
        const std::vector<TaggedSentence> &validation_sentences,
        const std::unordered_set<std::string> &training_vocabulary) override;

    void Validate(const std::vector<TaggedSentence> &tagged_sentences) const override;

    std::vector<std::string> Tag(const std::vector<std::string> &sentence) override;

    float ScoreTagging(const TaggedSentence &tagged_sentence) const override;

  public:
    LookupTable &word_table;
    LookupTable &tag_table;
    float p;
    std::mt19937 &generator;
    std::uniform_real_distribution<float> uniform, uniform_symmetric;
    int recurrent_state_dimension, tag_dimension, word_representation_dimension;
    Blob recurrent_state_input, recurrent_state_output;
    Blob classify_weights, classify_bias;
    Blob combine_weights, combine_bias;
    PartOfSpeechSentenceLayer part_of_speech_sentence;
    EuclideanLossLayer loss;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECURRENTNEURALNETWORKPARTOFSPEECHTAGGER_HPP_
