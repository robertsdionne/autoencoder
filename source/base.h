#ifndef AUTOENCODER_BASE_H_
#define AUTOENCODER_BASE_H_

/**
 * Allows us to save some typing when defining a purely abstract base class.
 */
#define DECLARE_INTERFACE(interface)\
public:\
  virtual ~interface() = default;\
  interface(const interface &) = delete;\
  interface &operator =(const interface &) = delete;\
protected:\
  interface() = default;\
private:

#endif  // AUTOENCODER_BASE_H_
