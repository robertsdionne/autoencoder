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

  template <typename F>
  class Evaluator;

  template <typename F>
  class LookupTable;

  template <typename F>
  class RecurrentNeuralNetworkPartOfSpeechTagger : public PartOfSpeechTagger<F> {
  public:
    RecurrentNeuralNetworkPartOfSpeechTagger(
      LookupTable<F> &word_table, LookupTable<F> &tag_table,
      F p, std::mt19937 &generator,
      int recurrent_state_dimension,
      int tag_dimension,
      int word_representation_dimension);

    virtual ~RecurrentNeuralNetworkPartOfSpeechTagger() = default;

    void ForwardCpu(const std::vector<std::string> &sentence, std::vector<std::string> *tags);

    F ForwardBackwardCpu(const TaggedSentence &tagged_sentence);

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

  public:
    LookupTable<F> &word_table;
    LookupTable<F> &tag_table;
    F p;
    std::mt19937 &generator;
    std::uniform_real_distribution<F> uniform, uniform_symmetric;
    int recurrent_state_dimension, tag_dimension, word_representation_dimension;
    Blob<F> recurrent_state_input, recurrent_state_output;
    Blob<F> classify_weights, classify_bias;
    Blob<F> combine_weights, combine_bias;
    PartOfSpeechSentenceLayer<F> part_of_speech_sentence;
    EuclideanLossLayer<F> loss;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECURRENTNEURALNETWORKPARTOFSPEECHTAGGER_HPP_
