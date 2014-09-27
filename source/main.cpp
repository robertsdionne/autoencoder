#include <gflags/gflags.h>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "blob.hpp"
#include "dataloader.hpp"
#include "lookuptable.hpp"
#include "recurrentneuralnetworkpartofspeechtagger.hpp"
#include "evaluator.hpp"

using namespace autoencoder;

using number = double;

namespace autoencoder {

  DEFINE_double(dropout_probability, 0.5, "the probability of masking out an input for dropout");

  DEFINE_int32(iterations, 100, "the number of training iterations");

  DEFINE_double(learning_rate, 1e-4, "the learning rate");

  DEFINE_double(momentum, 0.0, "the momentum");

  DEFINE_double(lambda_1, 1e-4, "the L1 regularization coefficient");

  DEFINE_double(lambda_2, 1e-4, "the L2 regularization coefficient");

  DEFINE_int32(random_seed, std::random_device()(), "seed the random number generator");

  DEFINE_int32(recurrent_state_dimension, 50, "the recurrent state dimension");

  DEFINE_bool(test, false, "whether to evaluate on the test data");

}  // namespace autoencoder

int main(int argument_count, char *arguments[]) {
  gflags::SetUsageMessage("Part-of-speech tagger implemented with a recurrent neural network.");
  gflags::ParseCommandLineFlags(&argument_count, &arguments, true);

  auto data_loader = DataLoader();

  std::cout << "Loading test sentences... ";
  std::cout.flush();
  auto test_sentences = data_loader.ReadTaggedSentences(
      FLAGS_test_filename, FLAGS_test_sentences);
  std::cout << "Done." << std::endl;

  std::cout << "Loading training sentences and vocabulary... ";
  std::cout.flush();
  auto training_sentences = data_loader.ReadTaggedSentences(
      FLAGS_training_filename, FLAGS_training_sentences);
  auto training_vocabulary = data_loader.FindVocabulary(training_sentences);
  std::cout << "Done." << std::endl;

  std::cout << "Loading in-domain validation sentences... ";
  std::cout.flush();
  auto validation_in_domain_sentences = data_loader.ReadTaggedSentences(
      FLAGS_validation_in_domain_filename);
  std::cout << "Done." << std::endl;

  std::cout << "Loading out-of-domain validation sentences... ";
  std::cout.flush();
  auto validation_out_of_domain_sentences = data_loader.ReadTaggedSentences(
      FLAGS_validation_out_of_domain_filename);
  std::cout << "Done." << std::endl;

  std::cout << "Loading overall tag set... ";
  std::cout.flush();
  auto tag_set = data_loader.FindTags({
    FLAGS_test_filename,
    FLAGS_training_filename,
    FLAGS_validation_in_domain_filename,
    FLAGS_validation_out_of_domain_filename,
  });
  auto tags = std::vector<std::string>(tag_set.begin(), tag_set.end());
  tags.insert(tags.begin(), "<START>");
  std::cout << "Done." << std::endl << std::endl;

  auto generator = std::mt19937(FLAGS_random_seed);
  auto word_table = LookupTable<number>::Load(
      generator, FLAGS_words_filename, FLAGS_vectors_filename);
  auto tag_vectors = std::vector<Blob<number>>(tags.size(), Blob<number>(tags.size()));
  for (auto i = 0; i < tags.size(); ++i) {
    tag_vectors.at(i).value(i) = number(1.0);
  }
  auto tag_table = LookupTable<number>(generator, tags, tag_vectors);
  auto part_of_speech_tagger = RecurrentNeuralNetworkPartOfSpeechTagger<number>(
      word_table, tag_table, FLAGS_dropout_probability, generator,
      FLAGS_recurrent_state_dimension, tags.size(),
      FLAGS_word_representation_dimension);
  auto evaluator = Evaluator<number>(generator);

  part_of_speech_tagger.Train(
      training_sentences, FLAGS_learning_rate, FLAGS_momentum, FLAGS_lambda_1, FLAGS_lambda_2,
      FLAGS_iterations, evaluator, validation_in_domain_sentences,
      training_vocabulary);

  std::cout << "Evaluating on training data... ";
  std::cout.flush();
  auto training_report = evaluator.Evaluate(
      part_of_speech_tagger, training_sentences, training_vocabulary);
  std::cout << "Done." << std::endl;
  std::cout << training_report << std::endl<< std::endl;

  std::cout << "Evaluating on in-domain validation data... ";
  std::cout.flush();
  auto validation_in_domain_report = evaluator.Evaluate(
      part_of_speech_tagger, validation_in_domain_sentences, training_vocabulary);
  std::cout << "Done." << std::endl;
  std::cout << validation_in_domain_report << std::endl<< std::endl;

  std::cout << "Evaluating on out-of-domain validation data... ";
  std::cout.flush();
  auto validation_out_of_domain_report = evaluator.Evaluate(
      part_of_speech_tagger, validation_out_of_domain_sentences, training_vocabulary);
  std::cout << "Done." << std::endl;
  std::cout << validation_out_of_domain_report << std::endl<< std::endl;

  if (FLAGS_test) {
    std::cout << "Evaluating on test data!!! ";
    std::cout.flush();
    auto test_report = evaluator.Evaluate(
        part_of_speech_tagger, test_sentences, training_vocabulary);
    std::cout << "Done!" << std::endl;
    std::cout << test_report << std::endl<< std::endl;
  }

  return 0;
}
