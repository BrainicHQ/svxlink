// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>
#include <iostream>

#include "opus_wrapper.h"

std::string opus::ErrorToString(int error) {
    switch (error) {
        case OPUS_OK:
            return "OK";
        case OPUS_BAD_ARG:
            return "One or more invalid/out of range arguments.";
        case OPUS_BUFFER_TOO_SMALL:
            return "The mode struct passed is invalid.";
        case OPUS_INTERNAL_ERROR:
            return "An internal error was detected.";
        case OPUS_INVALID_PACKET:
            return "The compressed data passed is corrupted.";
        case OPUS_UNIMPLEMENTED:
            return "Invalid/unsupported request number.";
        case OPUS_INVALID_STATE:
            return "An encoder or decoder structure is invalid or already freed.";
        default:
            return "Unknown error code: " + std::to_string(error);
    }
}

void opus::internal::OpusDestroyer::operator()(OpusEncoder* encoder) const
noexcept {
opus_encoder_destroy(encoder);
}

void opus::internal::OpusDestroyer::operator()(OpusDecoder* decoder) const
noexcept {
opus_decoder_destroy(decoder);
}

opus::Decoder::Decoder(opus_uint32 sample_rate, int num_channels)
        : num_channels_(num_channels) {
    int error{};
    decoder_.reset(opus_decoder_create(sample_rate, num_channels, &error));
    valid_ = error == OPUS_OK;
}

std::vector<opus_int16> opus::Decoder::Decode(
        const std::vector<unsigned char>& packet, int frame_size, bool decode_fec) {
    const auto frame_length = (frame_size * num_channels_ * sizeof(opus_int16));
    std::vector<opus_int16> decoded(frame_length);
    auto num_samples = opus_decode(decoder_.get(), packet.data(), packet.size(),
                                   decoded.data(), frame_size, decode_fec);
    if (num_samples < 0) {
        std::cout << "Decode error: " << opus::ErrorToString(num_samples);
        return {};
    }
    decoded.resize(num_samples * num_channels_);
    return decoded;
}