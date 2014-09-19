#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "evaluator.hpp"
#include "partofspeechtagger.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  template <typename F>
  EvaluationReport<F> Evaluator<F>::Evaluate(
      PartOfSpeechTagger<F> &part_of_speech_tagger,
      const std::vector<TaggedSentence> &tagged_sentences,
      const std::unordered_set<std::string> &training_vocabulary) const {
    auto number_of_tags = F(0.0);
    auto number_of_tags_correct = F(0.0);
    auto number_of_unknown_words = std::numeric_limits<F>::epsilon();
    auto number_of_unknown_words_correct = F(0.0);
    for (auto &tagged_sentence : tagged_sentences) {
      auto &sentence = tagged_sentence.words;
      auto &gold_tags = tagged_sentence.tags;
      auto guessed_tags = part_of_speech_tagger.Tag(sentence);
      for (auto i = 0; i < sentence.size(); ++i) {
        auto &word = sentence.at(i);
        auto &gold_tag = gold_tags.at(i);
        auto &guessed_tag = guessed_tags.at(i);
        if (guessed_tag == gold_tag) {
          number_of_tags_correct += F(1.0);
        }
        number_of_tags += F(1.0);
        if (training_vocabulary.cend() == training_vocabulary.find(word)) {
          if (guessed_tag == gold_tag) {
            number_of_unknown_words_correct += F(1.0);
          }
          number_of_unknown_words += F(1.0);
        }
      }
    }
    return EvaluationReport<F>{
      number_of_tags_correct / number_of_tags,
      number_of_unknown_words_correct / number_of_unknown_words
    };
  }

  template <typename F>
  std::ostream &operator <<(std::ostream &out, const EvaluationReport<F> &report) {
    out << "  Tag accuracy:     " << report.tag_accuracy << std::endl;
    out << "  Unknown accuracy: " << report.unknown_accuracy;
    return out;
  }

  template class Evaluator<float>;
  template class Evaluator<double>;

  template std::ostream &operator<<(std::ostream &out, const EvaluationReport<float> &report);
  template std::ostream &operator<<(std::ostream &out, const EvaluationReport<double> &report);

}  // namespace autoencoder
