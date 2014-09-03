#ifndef AUTOENCODER_TAGGEDSENTENCE_H_
#define AUTOENCODER_TAGGEDSENTENCE_H_

#include <cstdlib>
#include <string>
#include <vector>

namespace autoencoder {

  class TaggedSentence {
  public:
    TaggedSentence() = default;

    TaggedSentence(const std::vector<std::string> &words, const std::vector<std::string> &tags);

    virtual ~TaggedSentence() = default;

    size_t size() const;

  public:
    std::vector<std::string> words, tags;
  };

  std::string to_string(const TaggedSentence &sentence);

}  // namespace autoencoder

#endif  // AUTOENCODER_TAGGEDSENTENCE_H_
