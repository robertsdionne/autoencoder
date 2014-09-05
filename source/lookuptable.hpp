#ifndef AUTOENCODER_LOOKUPTABLE_HPP_
#define AUTOENCODER_LOOKUPTABLE_HPP_

#include <string>
#include <vector>
#include <unordered_map>

#include "blob.hpp"

namespace autoencoder {

  class LookupTable {
  public:
    LookupTable();

    LookupTable(const std::vector<std::string> &tokens, std::vector<Blob> &vectors);

    virtual ~LookupTable() = default;

    void ForwardCpu(const std::vector<std::string> &tokens, Blobs *top);

  private:
    std::unordered_map<std::string, int> token_indices;
    std::vector<Blob> *vectors;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LOOKUPTABLE_HPP_
