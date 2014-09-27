#include <limits>
#include <random>
#include <vector>
#include <unordered_set>

#include "blob.hpp"
#include "device.hpp"
#include "evaluator.hpp"
#include "lookuptable.hpp"
#include "recurrentneuralnetworkpartofspeechtagger.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  template<typename Distribution, typename Generator, typename F>
  void InitializeBlob(Distribution &distribution, Generator &generator, Blob<F> *blob) {
    for (auto i = 0; i < blob->height; ++i) {
      for (auto j = 0; j < blob->width; ++j) {
        blob->value(j, i) = distribution(generator);
      }
    }
  }

  template <typename F>
  RecurrentNeuralNetworkPartOfSpeechTagger<F>::RecurrentNeuralNetworkPartOfSpeechTagger(
      Device<F> &device,
      LookupTable<F> &word_table, LookupTable<F> &tag_table,
      F p, std::mt19937 &generator,
      int recurrent_state_dimension,
      int tag_dimension,
      int word_representation_dimension)
  : word_table(word_table), tag_table(tag_table),
    p(p), generator(generator),
    uniform(), uniform_symmetric(F(-0.1), F(0.1)),
    recurrent_state_dimension(recurrent_state_dimension),
    tag_dimension(tag_dimension),
    word_representation_dimension(word_representation_dimension),
    recurrent_state_input(recurrent_state_dimension),
    recurrent_state_output(recurrent_state_dimension),
    classify_weights(recurrent_state_dimension, tag_dimension), classify_bias(tag_dimension),
    combine_weights(
        recurrent_state_dimension + word_representation_dimension, recurrent_state_dimension),
    combine_bias(recurrent_state_dimension),
    part_of_speech_sentence(
        device, p, classify_weights, classify_bias, combine_weights, combine_bias, generator),
    loss(device) {}

  template <typename F>
  void RecurrentNeuralNetworkPartOfSpeechTagger<F>::ForwardCpu(
      const std::vector<std::string> &sentence, std::vector<std::string> *tags) {
    auto input = Blobs<F>{};
    word_table.ForwardCpu(sentence, &input);
    input.insert(input.begin(), &recurrent_state_input);

    auto guessed_tags = std::vector<Blob<F>>(sentence.size(), Blob<F>(tag_dimension));
    auto output = Blobs<F>{};
    for (auto &guessed_tag : guessed_tags) {
      output.push_back(&guessed_tag);
    }
    output.push_back(&recurrent_state_output);

    part_of_speech_sentence.ForwardCpu(Mode::kTest, input, &output);

    for (auto i = 0; i < output.size(); ++i) {
      output.at(i)->IsValid();
    }

    for (auto i = 1; i < guessed_tags.size(); ++i) {
      tags->push_back(tag_table.LookupToken(guessed_tags.at(i).values.Argmax()));
    }
  }

  template <typename F>
  F RecurrentNeuralNetworkPartOfSpeechTagger<F>::ForwardBackwardCpu(
      const TaggedSentence &tagged_sentence) {
    auto input = Blobs<F>{};
    word_table.ForwardCpu(tagged_sentence.words, &input);
    input.insert(input.begin(), &recurrent_state_input);

    auto target = Blobs<F>{};
    tag_table.ForwardCpu(tagged_sentence.tags, &target);

    auto guessed_tags = std::vector<Blob<F>>(tagged_sentence.size(), Blob<F>(tag_dimension));
    auto output = Blobs<F>{};
    for (auto &guessed_tag : guessed_tags) {
      output.push_back(&guessed_tag);
    }
    output.push_back(&recurrent_state_output);

    part_of_speech_sentence.ForwardCpu(Mode::kTrain, input, &output);

    auto output_and_target = Blobs<F>{};
    for (auto i = 0; i < target.size(); ++i) {
      output_and_target.push_back(output.at(i));
      output_and_target.push_back(target.at(i));
    }

    auto losses = std::vector<Blob<F>>(tagged_sentence.size(), Blob<F>(tag_dimension));
    auto loss_output = Blobs<F>{};
    for (auto &loss : losses) {
      loss_output.push_back(&loss);
    }

    auto result = loss.ForwardCpu(Mode::kTrain, output_and_target, &loss_output);

    for (auto &loss : losses) {
      loss.IsValid();
    }

    loss.BackwardCpu(loss_output, &output_and_target);
    part_of_speech_sentence.BackwardCpu(output, &input);

    return result;
  }

  template <typename F>
  void RecurrentNeuralNetworkPartOfSpeechTagger<F>::Train(
      const std::vector<TaggedSentence> &tagged_sentences,
      F learning_rate,
      F momentum,
      F lambda_1,
      F lambda_2,
      int iterations,
      Evaluator<F> &evaluator,
      const std::vector<TaggedSentence> &validation_sentences,
      const std::unordered_set<std::string> &training_vocabulary) {

    // InitializeBlob(uniform_symmetric, generator, &recurrent_state_input);
    InitializeBlob(uniform_symmetric, generator, &classify_weights);
    InitializeBlob(uniform_symmetric, generator, &combine_weights);

    std::uniform_int_distribution<int> uniform(0, tagged_sentences.size() - 1);

    constexpr auto minibatch_size = 100;

    std::cout << "Training... " << std::endl << std::endl;
    for (auto i = 0; i < iterations; ++i) {
      std::cout << "Evaluating on validation data... ";
      std::cout.flush();
      auto validation_report = evaluator.Evaluate(
          *this, validation_sentences, training_vocabulary);
      std::cout << "Done." << std::endl;
      std::cout << validation_report << std::endl<< std::endl;
      std::cout << "Starting iteration " << i << "... " << std::endl << std::endl;
      
      for (auto j = 0; j < tagged_sentences.size(); ++j) {
        // auto u = uniform(generator);
        auto u = j;
        ForwardBackwardCpu(tagged_sentences.at(u));

        classify_weights.ClipGradient(tagged_sentences.at(u).size());
        classify_bias.ClipGradient(tagged_sentences.at(u).size());
        combine_weights.ClipGradient(tagged_sentences.at(u).size());
        combine_bias.ClipGradient(tagged_sentences.at(u).size());

        classify_weights.L1Regularize(lambda_1);
        classify_bias.L1Regularize(lambda_1);
        combine_weights.L1Regularize(lambda_1);
        combine_bias.L1Regularize(lambda_1);

        auto magnitude = sqrt(classify_weights.values.SquareMagnitude()
            + classify_weights.values.SquareMagnitude()
            + combine_weights.values.SquareMagnitude()
            + combine_bias.values.SquareMagnitude());

        classify_weights.L2Regularize(lambda_2, magnitude);
        classify_bias.L2Regularize(lambda_2, magnitude);
        combine_weights.L2Regularize(lambda_2, magnitude);
        combine_bias.L2Regularize(lambda_2, magnitude);

        // auto difference_magnitude = sqrt(classify_weights.differences.SquareMagnitude()
        //     + classify_weights.differences.SquareMagnitude()
        //     + combine_weights.differences.SquareMagnitude()
        //     + combine_bias.differences.SquareMagnitude());

        // classify_weights.ClipGradient(difference_magnitude);
        // classify_bias.ClipGradient(difference_magnitude);
        // combine_weights.ClipGradient(difference_magnitude);
        // combine_bias.ClipGradient(difference_magnitude);

        // const auto modified_learning_rate = learning_rate * pow(F(0.1), i / 2.0);
        const auto modified_learning_rate = learning_rate;

        classify_weights.UpdateMomentum(modified_learning_rate, momentum);
        classify_bias.UpdateMomentum(modified_learning_rate, momentum);
        combine_weights.UpdateMomentum(modified_learning_rate, momentum);
        combine_bias.UpdateMomentum(modified_learning_rate, momentum);

        constexpr auto kAdaDeltaMemory = F(0.95);

        // classify_weights.UpdateAdaDelta(modified_learning_rate, kAdaDeltaMemory);
        // classify_bias.UpdateAdaDelta(modified_learning_rate, kAdaDeltaMemory);
        // combine_weights.UpdateAdaDelta(modified_learning_rate, kAdaDeltaMemory);
        // combine_bias.UpdateAdaDelta(modified_learning_rate, kAdaDeltaMemory);

        // classify_weights.UpdateAdaGrad(modified_learning_rate);
        // classify_bias.UpdateAdaGrad(modified_learning_rate);
        // combine_weights.UpdateAdaGrad(modified_learning_rate);
        // combine_bias.UpdateAdaGrad(modified_learning_rate);

        classify_weights.differences.Reset();
        classify_bias.differences.Reset();
        combine_weights.differences.Reset();
        combine_bias.differences.Reset();

        // if (j > 0 && j % 100 == 0) {
        //   std::cout << "Finished " << j << " sentences." << std::endl;
        // }

        if (j > 0 && j + 1 < tagged_sentences.size() && j % 1000 == 0) {
          // std::cout << std::endl << "Evaluating on validation data... ";
          // std::cout.flush();
          auto validation_report = evaluator.Evaluate(
              *this, validation_sentences, training_vocabulary);
          std::cout << "Done." << std::endl;
          std::cout << validation_report << std::endl << std::endl;
        }
      }
      std::cout << "Done." << std::endl << std::endl;
    }
    std::cout << "Done." << std::endl << std::endl;
  }

  template <typename F>
  void RecurrentNeuralNetworkPartOfSpeechTagger<F>::Validate(
      const std::vector<TaggedSentence> &tagged_sentences) const {
  }

  template <typename F>
  std::vector<std::string> RecurrentNeuralNetworkPartOfSpeechTagger<F>::Tag(
      const std::vector<std::string> &sentence) {
    auto tags = std::vector<std::string>();
    ForwardCpu(sentence, &tags);
    return tags;
  }

  template <typename F>
  F RecurrentNeuralNetworkPartOfSpeechTagger<F>::ScoreTagging(
      const TaggedSentence &tagged_sentence) const {
    return -std::numeric_limits<F>::infinity();
  }

  template class RecurrentNeuralNetworkPartOfSpeechTagger<float>;
  template class RecurrentNeuralNetworkPartOfSpeechTagger<double>;

}  // namespace autoencoder
