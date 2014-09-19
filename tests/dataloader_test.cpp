#include <gtest/gtest.h>

#include "dataloader.hpp"

using namespace autoencoder;

TEST(DataLoaderTest, TestLoadData) {
  auto data_loader = DataLoader();
  auto tagged_sentences = data_loader.ReadTaggedSentences(
      FLAGS_validation_in_domain_filename);

  EXPECT_EQ(1700, tagged_sentences.size());

  EXPECT_EQ(
      "influential_JJ members_NNS of_IN the_DT house_NNP ways_NNP and_CC means_NNP "
      "committee_NNP introduced_VBD legislation_NN that_WDT would_MD restrict_VB how_WRB the_DT "
      "new_JJ savings_NNS -_HYPH and_CC -_HYPH loan_NN bailout_NN agency_NN can_MD raise_VB "
      "capital_NN ,_, creating_VBG another_DT potential_JJ obstacle_NN to_IN the_DT government_NN "
      "\'s_POS sale_NN of_IN sick_JJ thrifts_NNS ._.",
      to_string(tagged_sentences.at(0)));
  EXPECT_EQ(
      "that_DT debt_NN would_MD be_VB paid_VBN off_RP as_IN the_DT assets_NNS are_VBP sold_VBN ,_, "
      "leaving_VBG the_DT total_JJ spending_NN for_IN the_DT bailout_NN at_IN $_$ 0_CD billion_CD "
      ",_, or_CC $_$ 0_CD billion_CD including_VBG interest_NN over_IN 0_CD years_NNS ._.",
      to_string(tagged_sentences.at(8)));

  auto tags = data_loader.FindTags({
    FLAGS_test_filename,
    FLAGS_training_filename,
    FLAGS_validation_in_domain_filename,
    FLAGS_validation_out_of_domain_filename,
  });

  EXPECT_EQ(47, tags.size());

  auto vocabulary = data_loader.FindVocabulary(tagged_sentences);

  EXPECT_EQ(5428, vocabulary.size());
}

TEST(DataLoaderTest, TestTokenizeNumbers) {
  auto data_loader = DataLoader();
  EXPECT_EQ(
      "0/0/0", data_loader.TokenizeNumbers("+123.456,999.123/-123.123,999,123/+333.9999,888"));
  EXPECT_EQ("0", data_loader.TokenizeNumbers("50,000"));
  EXPECT_EQ("0th", data_loader.TokenizeNumbers("-50000.000,0000th"));
}
