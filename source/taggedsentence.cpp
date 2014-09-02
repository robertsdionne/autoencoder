#include <cassert>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "taggedsentence.hpp"

namespace autoencoder {

  TaggedSentence::TaggedSentence(
      const std::vector<std::string> &words, const std::vector<std::string> &tags)
  : words(words), tags(tags) {
    assert(words.size() == tags.size());
  }

  size_t TaggedSentence::size() const {
    return words.size();
  }

  std::string to_string(const TaggedSentence &sentence) {
    std::ostringstream out;
    for (auto i = 0; i < sentence.words.size(); ++i) {
      out << sentence.words.at(i) << "_" << sentence.tags.at(i);
      if (i < sentence.words.size() - 1) {
        out << " ";
      }
    }
    return out.str();
  }

}  // namespace autoencoder
