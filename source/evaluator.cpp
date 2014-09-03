#include <iostream>
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
    return EvaluationReport();
  }

  std::ostream &operator <<(std::ostream &out, const EvaluationReport &report) {
    out << "\tTag accuracy: " << report.tag_accuracy << std::endl;
    out << "\tUnknown accuracy: " << report.unknown_accuracy;
    return out;
  }

}  // namespace autoencoder
