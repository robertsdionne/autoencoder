#include <iostream>

#include "audio.h"
#include "autoencoder.h"
#include "matrix.h"
#include "vector.h"

int main(int argument_count, char *arguments[]) {
  auto audio = autoencoder::ReadWavFile(u8"data/Major Lazer - Get Free.wav");
  delete [] audio.samples;
  autoencoder::Autoencoder ae{5, 10};
  autoencoder::Vector v(10);
  for (auto i = 0; i < 10; ++i) {
    v(i) = i;
  }
  auto result = ae.Forward(v);
  std::cout << result << std::endl;
  return 0;
}
