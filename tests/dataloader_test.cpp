#include <gtest/gtest.h>

#include "dataloader.hpp"

TEST(DataLoaderTest, TestLoadData) {
  auto data_loader = autoencoder::DataLoader();
  auto tagged_sentences = data_loader.ReadTaggedSentences(
      autoencoder::FLAGS_validation_in_domain_filename);
  EXPECT_EQ(1700, tagged_sentences.size());
  EXPECT_EQ(
      "influential_JJ members_NNS of_IN the_DT house_NNP ways_NNP and_CC means_NNP "
      "committee_NNP introduced_VBD legislation_NN that_WDT would_MD restrict_VB how_WRB the_DT "
      "new_JJ savings_NNS -_HYPH and_CC -_HYPH loan_NN bailout_NN agency_NN can_MD raise_VB "
      "capital_NN ,_, creating_VBG another_DT potential_JJ obstacle_NN to_IN the_DT government_NN "
      "\'s_POS sale_NN of_IN sick_JJ thrifts_NNS ._.",
      autoencoder::to_string(tagged_sentences.at(0)));
}

TEST(DataLoaderTest, TestTokenizeNumbers) {

}
