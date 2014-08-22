#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>

#include "audio.h"

namespace autoencoder {

  Audio ReadWavFile(const std::string &filename) {
    CFURLRef url = CFURLCreateWithBytes(
        kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(filename.c_str()), filename.size(),
        kCFStringEncodingUTF8, nullptr);
    AudioFileID audio_file;
    assert(noErr == AudioFileOpenURL(url, kAudioFileReadPermission, kAudioFileWAVEType, &audio_file));
    UInt32 format;
    UInt32 data_size = sizeof(format);
    assert(noErr == AudioFileGetProperty(
        audio_file, kAudioFilePropertyFileFormat, &data_size, &format));
    assert(kAudioFileWAVEType == format);
    AudioStreamBasicDescription description;
    data_size = sizeof(description);
    assert(noErr == AudioFileGetProperty(
        audio_file, kAudioFilePropertyDataFormat, &data_size, &description));
    assert(44100 == description.mSampleRate && 4 == description.mBytesPerPacket
        && 1 == description.mFramesPerPacket && 4 == description.mBytesPerFrame
        && 2 == description.mChannelsPerFrame && 16 == description.mBitsPerChannel);
    UInt64 packet_count;
    data_size = sizeof(packet_count);
    assert(noErr == AudioFileGetProperty(
        audio_file, kAudioFilePropertyAudioDataPacketCount, &data_size, &packet_count));
    UInt32 maximum_packet_size;
    data_size = sizeof(maximum_packet_size);
    assert(noErr == AudioFileGetProperty(
        audio_file, kAudioFilePropertyMaximumPacketSize, &data_size, &maximum_packet_size));
    assert(4 == maximum_packet_size);
    short *buffer = new short[packet_count * description.mChannelsPerFrame]();
    UInt32 bytes_read;
    UInt32 packets_to_read = packet_count;
    assert(noErr == AudioFileReadPackets(
        audio_file, false, &bytes_read, nullptr, 0, &packets_to_read, buffer));
    Audio result;
    result.samples = new float[packet_count * description.mChannelsPerFrame]();
    result.sample_count = packets_to_read;
    for (auto i = 0; i < 2 * packets_to_read; ++i) {
      result.samples[i] = static_cast<float>(buffer[i]) / std::numeric_limits<short>::max();
    }
    delete [] buffer;
    return result;
  }

}  // namespace autoencoder
