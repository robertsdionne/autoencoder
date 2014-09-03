#ifndef AUTOENCODER_INTERFACE_HPP_
#define AUTOENCODER_INTERFACE_HPP_

#define DECLARE_INTERFACE(interface)\
public:\
  virtual ~interface() = default;\
protected:\
  interface() = default;\
  interface(const interface &) = default;\
  interface &operator =(const interface &) = default;\
private:

#endif  // AUTOENCODER_INTERFACE_HPP_
