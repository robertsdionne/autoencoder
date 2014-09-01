#include <gflags/gflags.h>
#include <iostream>

#include "rectifiedlinearlayer.h"

int main(int argument_count, char *arguments[]) {
  gflags::SetUsageMessage("Part-of-speech tagger implemented with a recurrent neural network.");
  gflags::ParseCommandLineFlags(&argument_count, &arguments, true);
  return 0;
}
