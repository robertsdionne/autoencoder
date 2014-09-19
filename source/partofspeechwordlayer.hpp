#ifndef AUTOENCODER_PARTOFSPEECHWORDLAYER_HPP_
#define AUTOENCODER_PARTOFSPEECHWORDLAYER_HPP_

#include <random>

#include "blob.hpp"
#include "concatenatelayer.hpp"
#include "dropoutlayer.hpp"
#include "innerproductlayer.hpp"
#include "layer.hpp"
#include "rectifiedlinearlayer.hpp"
#include "softmaxlayer.hpp"

namespace autoencoder {

  class PartOfSpeechWordLayer : public Layer {
  public:
    PartOfSpeechWordLayer(
        float p,
        Blob<float> &classify_weights, Blob<float> &classify_bias,
        Blob<float> &combine_weights, Blob<float> &combine_bias,
        std::mt19937 &generator);

    virtual ~PartOfSpeechWordLayer() = default;

    float ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) override;

    void BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) override;

  private:
    DropoutLayer dropout;
    Blob<float> corrupted_recurrent, corrupted_word;
    InnerProductLayer classify;
    Blob<float> classified;
    SoftmaxLayer softmax;
    ConcatenateLayer concatenate;
    Blob<float> concatenated;
    InnerProductLayer combine;
    Blob<float> combined;
    RectifiedLinearLayer rectified_linear;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHWORDLAYER_HPP_
