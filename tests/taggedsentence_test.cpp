#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "taggedsentence.hpp"

using namespace autoencoder;

TEST(TaggedSentenceTest, TestConstruction) {
  auto words = std::vector<std::string>{"one", "two", "three"};
  auto tags = std::vector<std::string>{"TAG1", "TAG2", "TAG3"};
  auto sentence = TaggedSentence(words, tags);
  EXPECT_EQ(3, sentence.size());
  EXPECT_EQ("one_TAG1 two_TAG2 three_TAG3", to_string(sentence));
}

TEST(TaggedSentenceTest, TestConstructionMismatchedLength) {
  auto words = std::vector<std::string>{"one", "two"};
  auto tags = std::vector<std::string>{"TAG1"};
  EXPECT_DEATH(
      TaggedSentence(words, tags),
      "Assertion.*(failed)?.*words\\.size.*==.*tags\\.size.*(failed)?");
}
