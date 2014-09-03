#include <gflags/gflags.h>
#include <iostream>

#include "dataloader.hpp"

DEFINE_int32(iterations, 100, "the number of training iterations");
DEFINE_double(learning_rate, 0.01, "the learning rate");
DEFINE_int32(random_seed, -1, "seed the random number generator");
DEFINE_int32(recurrent_state_dimension, 50, "the recurrent state dimension");
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
  std::cout << "Done." << std::endl;

  return 0;
}
