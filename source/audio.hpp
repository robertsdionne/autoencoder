#ifndef AUTOENCODER_AUDIO_H_
#define AUTOENCODER_AUDIO_H_

#include <string>

namespace autoencoder {

  struct Audio {
    float *samples;
    long sample_count;
  };

  Audio ReadWavFile(const std::string &filename);

}  // namespace autoencoder

#endif  // AUTOENCODER_AUDIO_H_
