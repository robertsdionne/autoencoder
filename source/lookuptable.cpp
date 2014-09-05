#include <string>
#include <vector>

#include "lookuptable.hpp"

namespace autoencoder {

  LookupTable::LookupTable(const std::vector<std::string> &tokens, std::vector<Blob> &vectors)
  : token_indices(), vectors(vectors) {
    for (auto i = 0; i < tokens.size(); ++i) {
      token_indices.insert({tokens.at(i), i});
    }
  }

  void LookupTable::ForwardCpu(const std::vector<std::string> &tokens, Blobs *top) {
    top->clear();
    for (auto &token : tokens) {
      top->push_back(&vectors.at(token_indices.at(token)));
    }
  }

}  // namespace autoencoder
