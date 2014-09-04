#include <gflags/gflags.h>
#include <iostream>
#include <random>

#include "dataloader.hpp"
#include "recurrentneuralnetworkpartofspeechtagger.hpp"
#include "evaluator.hpp"

DEFINE_int32(iterations, 100, "the number of training iterations");
DEFINE_double(learning_rate, 0.01, "the learning rate");
DEFINE_int32(random_seed, std::random_device()(), "seed the random number generator");
DEFINE_int32(recurrent_state_dimension, 50, "the recurrent state dimension");
DEFINE_bool(test, false, "whether to evaluate on the test data");
DEFINE_int32(test_sentences, -1, "the number of test sentences to use");
DEFINE_int32(training_sentences, -1, "the number of training sentences to use");

int main(int argument_count, char *arguments[]) {
  gflags::SetUsageMessage("Part-of-speech tagger implemented with a recurrent neural network.");
  gflags::ParseCommandLineFlags(&argument_count, &arguments, true);

  auto data_loader = autoencoder::DataLoader();

  std::cout << "Loading test sentences... ";
  std::cout.flush();
  auto test_sentences = data_loader.ReadTaggedSentences(autoencoder::FLAGS_test_filename);
  std::cout << "Done." << std::endl;

  std::cout << "Loading training sentences and vocabulary... ";
  std::cout.flush();
  auto training_sentences = data_loader.ReadTaggedSentences(autoencoder::FLAGS_train_filename);
  auto training_vocabulary = data_loader.FindVocabulary(training_sentences);
  std::cout << "Done." << std::endl;

  std::cout << "Loading in-domain validation sentences... ";
  std::cout.flush();
  auto validation_in_domain_sentences = data_loader.ReadTaggedSentences(
      autoencoder::FLAGS_validation_in_domain_filename);
  std::cout << "Done." << std::endl;

  std::cout << "Loading out-of-domain validation sentences... ";
  std::cout.flush();
  auto validation_out_of_domain_sentences = data_loader.ReadTaggedSentences(
      autoencoder::FLAGS_validation_out_of_domain_filename);
  std::cout << "Done." << std::endl;

  std::cout << "Loading overall tag set... ";
  std::cout.flush();
  auto tags = data_loader.FindTags({
    autoencoder::FLAGS_test_filename,
    autoencoder::FLAGS_train_filename,
    autoencoder::FLAGS_validation_in_domain_filename,
    autoencoder::FLAGS_validation_out_of_domain_filename,
  });
  std::cout << "Done." << std::endl<< std::endl;

  auto part_of_speech_tagger = autoencoder::RecurrentNeuralNetworkPartOfSpeechTagger();
  auto evaluator = autoencoder::Evaluator();

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
