#include <algorithm>
#include <cassert>
#include <fstream>
#include <functional>
#include <gflags/gflags.h>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include "dataloader.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  DEFINE_string(test_filename, "data/en-web-test.tagged", "The test data.");

  DEFINE_string(train_filename, "data/en-wsj-train.pos", "The training data.");

  DEFINE_string(
      validation_in_domain_filename, "data/en-wsj-dev.pos", "The in-domain validation data.");

  DEFINE_string(validation_out_of_domain_filename,
      "data/en-web-weblogs-dev.pos", "The out-of-domain validation data.");

    std::vector<TaggedSentence> DataLoader::ReadTaggedSentences(const std::string &filename) const {
      std::ifstream in(filename);
      assert(in);
      auto locale = std::locale("en_US.utf8");
      auto tagged_sentences = std::vector<TaggedSentence>();
      auto words = std::vector<std::string>();
      auto tags = std::vector<std::string>();
      auto line = std::string();
      while (std::getline(in, line)) {
        if (line.size() > 0) {
          std::istringstream line_in(line);
          auto word = std::string();
          auto tag = std::string();
          line_in >> word >> tag;
          std::transform(word.begin(), word.end(), word.begin(), [&locale] (char c) {
            return std::tolower(c, locale);
          });
          words.push_back(word);
          tags.push_back(tag);
        } else {
          tagged_sentences.push_back(TaggedSentence(words, tags));
          words.clear();
          tags.clear();
        }
      }
      return tagged_sentences;
    }

}  // namespace autoencoder
