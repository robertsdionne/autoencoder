#include <limits>
#include <random>
#include <vector>
#include <unordered_set>

#include "blob.hpp"
#include "evaluator.hpp"
#include "lookuptable.hpp"
#include "recurrentneuralnetworkpartofspeechtagger.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  template<typename Distribution, typename Generator>
  void InitializeBlob(Distribution &distribution, Generator &generator, autoencoder::Blob *blob) {
    for (auto i = 0; i < blob->height; ++i) {
      for (auto j = 0; j < blob->width; ++j) {
        blob->value(j, i) = distribution(generator);
      }
    }
  }

  RecurrentNeuralNetworkPartOfSpeechTagger::RecurrentNeuralNetworkPartOfSpeechTagger(
      LookupTable &word_table, LookupTable &tag_table,
      float p, std::mt19937 &generator,
      int recurrent_state_dimension,
      int tag_dimension,
      int word_representation_dimension)
  : word_table(word_table), tag_table(tag_table),
    p(p), generator(generator),
    uniform(), uniform_symmetric(-0.1f, 0.1f),
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
        p, classify_weights, classify_bias, combine_weights, combine_bias, generator),
    loss() {}

  void RecurrentNeuralNetworkPartOfSpeechTagger::ForwardCpu(
      const std::vector<std::string> &sentence, std::vector<std::string> *tags) {
    auto input = Blobs{};
    word_table.ForwardCpu(sentence, &input);
    input.insert(input.begin(), &recurrent_state_input);

    auto guessed_tags = std::vector<Blob>(sentence.size(), Blob(tag_dimension));
    auto output = Blobs{};
    for (auto &guessed_tag : guessed_tags) {
      output.push_back(&guessed_tag);
    }
    output.push_back(&recurrent_state_output);

    part_of_speech_sentence.ForwardCpu(input, &output);

    for (auto i = 0; i < output.size(); ++i) {
      output.at(i)->IsValid();
    }

    for (auto &guessed_tag : guessed_tags) {
      tags->push_back(tag_table.LookupToken(guessed_tag.values.Argmax()));
    }
  }

  void RecurrentNeuralNetworkPartOfSpeechTagger::ForwardBackwardCpu(
      const TaggedSentence &tagged_sentence) {
    // std::cout << to_string(tagged_sentence) << std::endl;
    auto input = Blobs{};
    word_table.ForwardCpu(tagged_sentence.words, &input);
    input.insert(input.begin(), &recurrent_state_input);

    // std::cout << "input.size " << input.size() << std::endl;
    // for (auto i = 1; i < input.size(); ++i) {
    //   std::cout << tagged_sentence.words.at(i - 1) << "_" << tagged_sentence.tags.at(i - 1) << std::endl;
    //   std::cout << "known " << word_table.known(tagged_sentence.words.at(i - 1)) << std::endl;
    //   std::cout << "known index " << word_table.known_index(tagged_sentence.words.at(i - 1)) << std::endl;
    //   std::cout << "unknown " << word_table.unknown(tagged_sentence.words.at(i - 1)) << std::endl;
    //   std::cout << "unknown index " << word_table.unknown_index(tagged_sentence.words.at(i - 1)) << std::endl;
    //   std::cout << "input.at(i) " << input.at(i) << std::endl;
    //   std::cout << "input.at(i).width " << input.at(i)->width << std::endl;
    //   std::cout << "input.at(i).values.width " << input.at(i)->values.width << std::endl;
    //   std::cout << "input.at(i).differences.width " << input.at(i)->differences.width << std::endl;
    //   std::cout << "input.at(i).values.size " << input.at(i)->values.values.size() << std::endl;
    //   std::cout << "input.at(i).differences.size " << input.at(i)->differences.values.size() << std::endl;
    //   std::cout << "i " << i << std::endl;
    //   std::cout << std::endl;
    // }

    auto target = Blobs{};
    tag_table.ForwardCpu(tagged_sentence.tags, &target);

    auto guessed_tags = std::vector<Blob>(tagged_sentence.size(), Blob(tag_dimension));
    auto output = Blobs{};
    for (auto &guessed_tag : guessed_tags) {
      output.push_back(&guessed_tag);
    }
    output.push_back(&recurrent_state_output);

    part_of_speech_sentence.ForwardCpu(input, &output);

    auto output_and_target = Blobs{};
    for (auto i = 0; i < target.size(); ++i) {
      output_and_target.push_back(output.at(i));
      output_and_target.push_back(target.at(i));
    }

    auto losses = std::vector<Blob>(tagged_sentence.size(), Blob(tag_dimension));
    auto loss_output = Blobs{};
    for (auto &loss : losses) {
      loss_output.push_back(&loss);
    }

    loss.ForwardCpu(output_and_target, &loss_output);

    for (auto &loss : losses) {
      loss.IsValid();
    }

    loss.BackwardCpu(loss_output, &output_and_target);
    part_of_speech_sentence.BackwardCpu(output, &input);
  }

  void RecurrentNeuralNetworkPartOfSpeechTagger::Train(
      const std::vector<TaggedSentence> &tagged_sentences,
      float learning_rate,
      int iterations,
      Evaluator &evaluator,
      const std::vector<TaggedSentence> &validation_sentences,
      const std::unordered_set<std::string> &training_vocabulary) {

    InitializeBlob(uniform_symmetric, generator, &recurrent_state_input);
    InitializeBlob(uniform_symmetric, generator, &classify_weights);
    InitializeBlob(uniform_symmetric, generator, &combine_weights);

    std::cout << "Training... " << std::endl;
    for (auto i = 0; i < iterations; ++i) {
      std::cout << "Starting iteration " << i << "... " << std::endl << std::endl;
      std::cout << "Evaluating on validation data... ";
      std::cout.flush();
      auto validation_report = evaluator.Evaluate(*this, validation_sentences, training_vocabulary);
      std::cout << "Done." << std::endl;
      std::cout << validation_report << std::endl<< std::endl;
      // std::cout << classify_weights.values << std::endl;
      // std::cout << classify_bias.values << std::endl;
      // std::cout << combine_weights.values << std::endl;
      // std::cout << combine_bias.values << std::endl;
      
      for (auto j = 0; j < tagged_sentences.size(); ++j) {
        if (j > 0 && j % 100 == 0) {
          std::cout << "Finished " << j << " sentences." << std::endl;
          // std::cout << to_string(tagged_sentences.at(j)) << std::endl;
        }
        ForwardBackwardCpu(tagged_sentences.at(j));

        recurrent_state_input.Update(learning_rate);
        classify_weights.Update(learning_rate);
        classify_bias.Update(learning_rate);
        combine_weights.Update(learning_rate);
        combine_bias.Update(learning_rate);
      }
      std::cout << "Done." << std::endl << std::endl << std::endl;
    }
    std::cout << "Done." << std::endl << std::endl;
  }

  void RecurrentNeuralNetworkPartOfSpeechTagger::Validate(
      const std::vector<TaggedSentence> &tagged_sentences) const {
  }

  std::vector<std::string> RecurrentNeuralNetworkPartOfSpeechTagger::Tag(
      const std::vector<std::string> &sentence) {
    auto tags = std::vector<std::string>();
    ForwardCpu(sentence, &tags);
    return tags;
  }

  float RecurrentNeuralNetworkPartOfSpeechTagger::ScoreTagging(
      const TaggedSentence &tagged_sentence) const {
    return -std::numeric_limits<float>::infinity();
  }

}  // namespace autoencoder
