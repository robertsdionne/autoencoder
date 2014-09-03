#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

#include "dumbpartofspeechtagger.hpp"
#include "taggedsentence.hpp"

TEST(DumbPartOfSpeechTaggerTest, TestTag) {
  auto pos_tagger = autoencoder::DumbPartOfSpeechTagger();
  auto words = std::vector<std::string>{"one", "two", "three"};
  auto tags = pos_tagger.Tag(words);
  EXPECT_EQ(3, tags.size());
  for (auto &tag : tags) {
    EXPECT_EQ("NN", tag);
  }
}

TEST(DumbPartOfSpeechTaggerTest, TestScoreTagging) {
  auto pos_tagger = autoencoder::DumbPartOfSpeechTagger();
  auto tagging = autoencoder::TaggedSentence();
  EXPECT_EQ(
      -std::numeric_limits<float>::infinity(), pos_tagger.ScoreTagging(tagging));
}
