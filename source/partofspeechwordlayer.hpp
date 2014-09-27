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

  template <typename F> class Device;

  template <typename F>
  class PartOfSpeechWordLayer : public Layer<F> {
  public:
    PartOfSpeechWordLayer(
        Device<F> &device,
        F p,
        Blob<F> &classify_weights, Blob<F> &classify_bias,
        Blob<F> &combine_weights, Blob<F> &combine_bias,
        std::mt19937 &generator);

    virtual ~PartOfSpeechWordLayer() = default;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override;

  private:
    DropoutLayer<F> dropout;
    Blob<F> corrupted_recurrent, corrupted_word;
    InnerProductLayer<F> classify;
    Blob<F> classified;
    SoftmaxLayer<F> softmax;
    ConcatenateLayer<F> concatenate;
    Blob<F> concatenated;
    InnerProductLayer<F> combine;
    Blob<F> combined;
    RectifiedLinearLayer<F> rectified_linear;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHWORDLAYER_HPP_
