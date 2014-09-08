#include <algorithm>
#include <cassert>
#include <fstream>
#include <functional>
#include <gflags/gflags.h>
#include <initializer_list>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "dataloader.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  DEFINE_string(test_filename, "data/en-web-test.tagged", "The test data.");

  DEFINE_int32(test_sentences, -1, "the number of test sentences to use");

  DEFINE_string(training_filename, "data/en-wsj-train.pos", "The training data.");

  DEFINE_int32(training_sentences, -1, "the number of training sentences to use");

  DEFINE_string(
      validation_in_domain_filename, "data/en-wsj-dev.pos", "The in-domain validation data.");

  DEFINE_string(validation_out_of_domain_filename,
      "data/en-web-weblogs-dev.pos", "The out-of-domain validation data.");

  std::set<std::string> DataLoader::FindTags(
      const std::initializer_list<std::string> &filenames) const {
    auto tags = std::set<std::string>();
    for (auto &filename : filenames) {
      std::ifstream in(filename);
      assert(in);
      auto line = std::string();
      while (std::getline(in, line)) {
        if (line.size() > 0) {
          std::istringstream line_in(line);
          auto word = std::string();
          auto tag = std::string();
          line_in >> word >> tag;
          tags.insert(tag);
        }
      }
    }
    return tags;
  }

  std::set<std::string> DataLoader::FindTags(const std::vector<TaggedSentence> &sentences) const {
    auto tags = std::set<std::string>();
    for (auto &sentence : sentences) {
      for (auto &tag : sentence.tags) {
        tags.insert(tag);
      }
    }
    return tags;
  }

  std::unordered_set<std::string> DataLoader::FindVocabulary(
      const std::vector<TaggedSentence> &sentences) const {
    auto vocabulary = std::unordered_set<std::string>();
    for (auto &sentence : sentences) {
      for (auto &word : sentence.words) {
        vocabulary.insert(word);
      }
    }
    return vocabulary;
  }

  std::vector<TaggedSentence> DataLoader::ReadTaggedSentences(
      const std::string &filename, long amount) const {
    std::ifstream in(filename);
    assert(in);
    auto tagged_sentences = std::vector<TaggedSentence>();
    auto words = std::vector<std::string>();
    auto tags = std::vector<std::string>();
    auto line = std::string();
    auto sentences = 0L;
    while (std::getline(in, line)) {
      if (line.size() > 0) {
        std::istringstream line_in(line);
        auto word = std::string();
        auto tag = std::string();
        line_in >> word >> tag;

        // Make word lower case.
        std::transform(word.begin(), word.end(), word.begin(), [] (char c) {
          return std::tolower(c);
        });

        // Replace numbers with 0.
        word = TokenizeNumbers(word);

        words.push_back(word);
        tags.push_back(tag);
      } else {
        tagged_sentences.push_back(TaggedSentence(words, tags));
        words.clear();
        tags.clear();
        sentences += 1;
      }
      if (amount > -1 && sentences >= amount) {
        break;
      }
    }
    return tagged_sentences;
  }

  std::string DataLoader::TokenizeNumbers(const std::string &input) const {
    auto number_pattern = std::regex("[\\+\\-]?[0-9]+([\\.,][0-9]*)*");
    return std::regex_replace(input, number_pattern, "0");
  }

}  // namespace autoencoder
