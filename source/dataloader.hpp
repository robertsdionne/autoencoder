#ifndef AUTOENCODER_DATALOADER_HPP_
#define AUTOENCODER_DATALOADER_HPP_

#include <gflags/gflags.h>
#include <initializer_list>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "taggedsentence.hpp"

namespace autoencoder {

  DECLARE_string(test_filename);

  DECLARE_int32(test_sentences);

  DECLARE_string(training_filename);

  DECLARE_int32(training_sentences);
  
  DECLARE_string(validation_in_domain_filename);
  
  DECLARE_string(validation_out_of_domain_filename);

  class DataLoader {
  public:
    DataLoader() = default;

    virtual ~DataLoader() = default;

    std::set<std::string> FindTags(const std::initializer_list<std::string> &filenames) const;

    std::set<std::string> FindTags(const std::vector<TaggedSentence> &sentences) const;

    std::unordered_set<std::string> FindVocabulary(
        const std::vector<TaggedSentence> &sentences) const;

    std::vector<TaggedSentence> ReadTaggedSentences(
        const std::string &filename, long amount = -1) const;

    std::string TokenizeNumbers(const std::string &input) const;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_DATALOADER_HPP_
