#include <gflags/gflags.h>
#include <iostream>

#include "rectifiedlinearlayer.h"

constexpr int kWordVectorDimension = 50;

DEFINE_int32(iterations, 100, "the number of training iterations");
DEFINE_double(learning_rate, 0.01, "the learning rate");
DEFINE_int32(random_seed, 12345, "seed the random number generator");
DEFINE_int32(recurrent_state_dimension, kWordVectorDimension, "the recurrent state dimension");
DEFINE_int32(test_sentences, -1, "the number of test sentences to use");
DEFINE_int32(training_sentences, -1, "the number of training sentences to use");

int main(int argument_count, char *arguments[]) {
  gflags::SetUsageMessage("Part-of-speech tagger implemented with a recurrent neural network.");
  gflags::ParseCommandLineFlags(&argument_count, &arguments, true);
  return 0;
}
