#include <limits>
#include <string>
#include <vector>

#include "dumbpartofspeechtagger.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {  

  void DumbPartOfSpeechTagger::Train(
      const std::vector<TaggedSentence> &tagged_sentences,
      float learning_rate,
      int iterations) {
  }

  void DumbPartOfSpeechTagger::Validate(const std::vector<TaggedSentence> &tagged_sentences) const {
  }

  std::vector<std::string> DumbPartOfSpeechTagger::Tag(
      const std::vector<std::string> &sentence) const {
    return std::vector<std::string>();
  }

  float DumbPartOfSpeechTagger::ScoreTagging(const TaggedSentence &tagged_sentence) const {
    return -std::numeric_limits<float>::infinity();
  }

}  // namespace autoencoder
