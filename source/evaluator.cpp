#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "evaluator.hpp"
#include "partofspeechtagger.hpp"
#include "taggedsentence.hpp"

namespace autoencoder {

  template <typename F>
  Evaluator<F>::Evaluator(std::mt19937 &generator) : generator(generator) {}

  template <typename F>
  EvaluationReport<F> Evaluator<F>::Evaluate(
      PartOfSpeechTagger<F> &part_of_speech_tagger,
      const std::vector<TaggedSentence> &tagged_sentences,
      const std::unordered_set<std::string> &training_vocabulary) {
    auto number_of_tags = F(0.0);
    auto number_of_tags_correct = F(0.0);
    auto number_of_unknown_words = std::numeric_limits<F>::epsilon();
    auto number_of_unknown_words_correct = std::numeric_limits<F>::epsilon();
    auto report = EvaluationReport<F>{};
    // for (auto &tagged_sentence : tagged_sentences) {
    std::uniform_int_distribution<int> uniform(0, tagged_sentences.size() - 1);
    auto sentence_count = std::min(1000UL, tagged_sentences.size());
    for (auto j = 0; j < sentence_count; ++j) {
      auto &tagged_sentence = tagged_sentences.at(uniform(generator));
      auto &sentence = tagged_sentence.words;
      auto &gold_tags = tagged_sentence.tags;
      auto guessed_tags = part_of_speech_tagger.Tag(sentence);
      // HORRIBLE HACK: skipping the last tag since the tagger currently doesn't tag the last word.
      for (auto i = 0; i < sentence.size() - 1; ++i) {
        auto &word = sentence.at(i);
        auto &gold_tag = gold_tags.at(i);
        auto &guessed_tag = guessed_tags.at(i);
        if (guessed_tag == gold_tag) {
          number_of_tags_correct += F(1.0);
          report.histogram[i] += F(1.0) / sentence_count;
        }
        report.length_histogram[i] += F(1.0) / sentence_count;
        number_of_tags += F(1.0);
        if (training_vocabulary.cend() == training_vocabulary.find(word)) {
          if (guessed_tag == gold_tag) {
            number_of_unknown_words_correct += F(1.0);
          }
          number_of_unknown_words += F(1.0);
        }
      }
    }
    report.tag_accuracy = number_of_tags_correct / number_of_tags;
    report.tag_accuracy_delta = report.tag_accuracy - previous_tag_accuracy;
    report.unknown_accuracy = number_of_unknown_words_correct / number_of_unknown_words;
    report.unknown_accuracy_delta = report.unknown_accuracy - previous_unknown_accuracy;
    previous_tag_accuracy = report.tag_accuracy;
    previous_unknown_accuracy = report.unknown_accuracy;
    return report;
  }

  template <typename F>
  std::ostream &operator <<(std::ostream &out, EvaluationReport<F> &report) {
    auto max_index = 0;
    for (auto entry : report.histogram) {
      max_index = std::max(max_index, entry.first);
    }
    const auto kGrid = std::string{"---+---|---+---|---+---|---+---|---+---|---+---|---+---|---+---|---+---|---+---|"};
    for (auto i = max_index - 1; i >= 0; --i) {
      auto amount = F(80.0) * report.histogram[i];
      auto max_amount = F(80.0) * report.length_histogram[i];
      for (auto j = 0; j < amount; ++j) {
        out << "#";
      }
      for (auto j = ceil(amount); j < max_amount - 1; ++j) {
        out << " ";
      }
      if (ceil(amount) < max_amount) {
        out << ".";
      }
      if (0 < i && 0 == i % 10) {
        out << kGrid.substr(ceil(max_amount));
      }
      out << std::endl;
    }
    out << kGrid << std::endl;
    out << "  Tag accuracy:     " << report.tag_accuracy << "\t\u0394: " << report.tag_accuracy_delta << std::endl;
    out << "  Unknown accuracy: " << report.unknown_accuracy << "\t\u0394: " << report.unknown_accuracy_delta;
    return out;
  }

  template class Evaluator<float>;
  template class Evaluator<double>;

  template std::ostream &operator<<(std::ostream &out, EvaluationReport<float> &report);
  template std::ostream &operator<<(std::ostream &out, EvaluationReport<double> &report);

}  // namespace autoencoder
