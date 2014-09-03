#ifndef AUTOENCODER_AUDIO_HPP_
#define AUTOENCODER_AUDIO_HPP_

#include <string>

namespace autoencoder {

  struct Audio {
    float *samples;
    long sample_count;
  };

  Audio ReadWavFile(const std::string &filename);

}  // namespace autoencoder

#endif  // AUTOENCODER_AUDIO_HPP_
