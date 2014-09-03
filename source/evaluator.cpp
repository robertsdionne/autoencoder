#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "evaluator.hpp"
#include "partofspeechtagger.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  EvaluationReport Evaluator::Evaluate(
      PartOfSpeechTagger &pos_tagger,
      const std::vector<TaggedSentence> &tagged_sentences,
      const std::unordered_set<std::string> &training_vocabulary) const {
    auto number_of_tags = 0.0f;
    auto number_of_tags_correct = 0.0f;
    auto number_of_unknown_words = std::numeric_limits<float>::epsilon();
    auto number_of_unknown_words_correct = 0.0f;
    for (auto &tagged_sentence : tagged_sentences) {
      auto &sentence = tagged_sentence.words;
      auto &gold_tags = tagged_sentence.tags;
      auto guessed_tags = pos_tagger.Tag(sentence);
      for (auto i = 0; i < sentence.size(); ++i) {
        auto &word = sentence.at(i);
        auto &gold_tag = gold_tags.at(i);
        auto &guessed_tag = guessed_tags.at(i);
        if (guessed_tag == gold_tag) {
          number_of_tags_correct += 1.0f;
        }
        number_of_tags += 1.0f;
        if (training_vocabulary.cend() == training_vocabulary.find(word)) {
          if (guessed_tag == gold_tag) {
            number_of_unknown_words_correct += 1.0f;
          }
          number_of_unknown_words += 1.0f;
        }
      }
    }
    return EvaluationReport{
      number_of_tags_correct / number_of_tags,
      number_of_unknown_words_correct / number_of_unknown_words
    };
  }

  std::ostream &operator <<(std::ostream &out, const EvaluationReport &report) {
    out << "  Tag accuracy:     " << report.tag_accuracy << std::endl;
    out << "  Unknown accuracy: " << report.unknown_accuracy;
    return out;
  }

}  // namespace autoencoder
