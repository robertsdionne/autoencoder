#include <cassert>
#include <fstream>
#include <gflags/gflags.h>
#include <sstream>
#include <string>
#include <vector>

#include "blob.hpp"
#include "lookuptable.hpp"

namespace autoencoder {

  DEFINE_string(words_filename, "embeddings/words.lst", "The words.");

  DEFINE_string(vectors_filename, "embeddings/embeddings.txt", "The vectors.");

  DEFINE_int32(word_representation_dimension, 50, "the word representation dimension");
  
  template <typename F>
  LookupTable<F>::LookupTable(std::mt19937 &generator)
  : generator(generator), uniform(F(-1.0), F(1.0)),
    token_indices(), unknown_token_indices(),
    vectors(), unknown_vectors(),
    index_tokens(), unknown_index_tokens() {}

  template <typename F>
  LookupTable<F>::LookupTable(
      std::mt19937 &generator,
      const std::vector<std::string> &tokens, const std::vector<Blob<F>> &vectors)
  : generator(generator),
    token_indices(), unknown_token_indices(),
    vectors(vectors), unknown_vectors(),
    index_tokens(), unknown_index_tokens() {
    for (auto i = 0; i < tokens.size(); ++i) {
      token_indices.insert({tokens.at(i), i});
      index_tokens.insert({i, tokens.at(i)});
      vectors.at(i).IsValid();
    }
  }

  template <typename F>
  void LookupTable<F>::ForwardCpu(const std::vector<std::string> &tokens, Blobs<F> *top) {
    top->clear();
    for (auto &token : tokens) {
      if (token_indices.cend() == token_indices.find(token)) {
        // if (unknown_token_indices.cend() == unknown_token_indices.find(token)) {
        //   assert(FLAGS_word_representation_dimension == 50);
        //   unknown_token_indices.insert({token, unknown_vectors.size()});
        //   unknown_vectors.push_back(Blob(FLAGS_word_representation_dimension));
        //   for (auto i = 0; i < FLAGS_word_representation_dimension; ++i) {
        //     unknown_vectors.back().value(i) = uniform(generator);
        //   }
        // }
        // assert(unknown_vectors.at(unknown_token_indices.at(token)).values.values.size() == 50);
        // top->push_back(&unknown_vectors.at(unknown_token_indices.at(token)));
        top->push_back(&vectors.at(token_indices.at("UNKNOWN")));
      } else {
        // if (vectors.at(token_indices.at(token)).values.values.size() != 50) {
        //   std::cout << "assertion " << vectors.at(token_indices.at(token)).values.values.size() << std::endl;
        // }
        top->push_back(&vectors.at(token_indices.at(token)));
      }
      top->back()->IsValid();
    }
  }

  template <typename F>
  std::string LookupTable<F>::LookupToken(int index) {
    if (index_tokens.cend() == index_tokens.find(index)) {
      return "?";
    } else {
      return index_tokens.at(index);
    }
  }

  template <typename F>
  LookupTable<F> LookupTable<F>::Load(
      std::mt19937 &generator,
      const std::string &words_filename, const std::string &vectors_filename) {
    auto words = std::vector<std::string>();
    std::ifstream words_in(words_filename);
    assert(words_in);
    auto line = std::string();
    while (std::getline(words_in, line)) {
      if (line.size() > 0) {
        std::istringstream line_in(line);
        auto word = std::string();
        line_in >> word;
        words.push_back(word);
      }
    }

    auto vectors = std::vector<Blob<F>>();
    std::ifstream vectors_in(vectors_filename);
    assert(vectors_in);
    while (std::getline(vectors_in, line)) {
      if (line.size() > 0) {
        std::istringstream line_in(line);
        vectors.push_back(Blob<F>(FLAGS_word_representation_dimension));
        for (auto i = 0; i < FLAGS_word_representation_dimension; ++i) {
          F value = 0.0f;
          line_in >> value;
          vectors.back().value(i) = value;
        }
      }
    }
    return LookupTable(generator, words, vectors);
  }

  template class LookupTable<float>;
  template class LookupTable<double>;

}  // namespace autoencoder
