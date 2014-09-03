#ifndef AUTOENCODER_DATALOADER_H_
#define AUTOENCODER_DATALOADER_H_

#include <gflags/gflags.h>
#include <vector>
#include <string>

#include "taggedsentence.hpp"

namespace autoencoder {

  DECLARE_string(test_filename);

  DECLARE_string(train_filename);
  
  DECLARE_string(validation_in_domain_filename);
  
  DECLARE_string(validation_out_of_domain_filename);

  class DataLoader {
  public:
    DataLoader() = default;

    virtual ~DataLoader() = default;

    std::vector<TaggedSentence> ReadTaggedSentences(const std::string &filename) const;

    std::string TokenizeNumbers(const std::string &input) const;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_DATALOADER_H_
