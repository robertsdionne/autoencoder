#ifndef AUTOENCODER_LOOKUPTABLE_HPP_
#define AUTOENCODER_LOOKUPTABLE_HPP_

#include <gflags/gflags.h>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include "blob.hpp"

namespace autoencoder {

  DECLARE_string(words_filename);

  DECLARE_string(vectors_filename);

  DECLARE_int32(word_representation_dimension);

  template <typename F>
  class LookupTable {
  public:
    LookupTable(std::mt19937 &generator);

    LookupTable(std::mt19937 &generator,
        const std::vector<std::string> &tokens, const std::vector<Blob<F>> &vectors);

    virtual ~LookupTable() = default;

    bool known(const std::string &word) {
      return token_indices.cend() != token_indices.find(word);
    }

    int known_index(const std::string &word) {
      if (known(word)) {
        return token_indices.at(word);
      } else {
        return -1;
      }
    }

    bool unknown(const std::string &word) {
      return unknown_token_indices.cend() != unknown_token_indices.find(word);
    }

    int unknown_index(const std::string &word) {
      if (unknown(word)) {
        return unknown_token_indices.at(word);
      } else {
        return -1;
      }
    }

    void ForwardCpu(const std::vector<std::string> &tokens, Blobs<F> *top);

    std::string LookupToken(int index);

    static LookupTable<F> Load(
        std::mt19937 &generator,
        const std::string &words_filename, const std::string &vectors_filename);

  private:
    std::mt19937 &generator;
    std::uniform_real_distribution<F> uniform;
    std::unordered_map<std::string, int> token_indices, unknown_token_indices;
    std::vector<Blob<F>> vectors, unknown_vectors;
    std::unordered_map<int, std::string> index_tokens, unknown_index_tokens;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LOOKUPTABLE_HPP_
