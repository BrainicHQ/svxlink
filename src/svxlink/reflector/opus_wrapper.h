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

#ifndef OPUSCPP_OPUS_WRAPPER_H_
#define OPUSCPP_OPUS_WRAPPER_H_

#include <memory>
#include <string>
#include <vector>

#include "opus/opus.h"

namespace opus {

    std::string ErrorToString(int error);

    namespace internal {
// Deleter for OpusEncoders and OpusDecoders
        struct OpusDestroyer {
            void operator()(OpusEncoder* encoder) const noexcept;
            void operator()(OpusDecoder* decoder) const noexcept;
        };
        template <typename T>
        using opus_uptr = std::unique_ptr<T, OpusDestroyer>;
    }  // namespace internal

    class Decoder {
    public:
        // see documentation at:
        // https://mf4.xiph.org/jenkins/view/opus/job/opus/ws/doc/html/group__opus__decoder.html#ga753f6fe0b699c81cfd47d70c8e15a0bd
        // Fs corresponds to sample_rate
        Decoder(opus_uint32 sample_rate, int num_channels);

        // Takes a sequence of encoded packets and decodes them. Returns the decoded
        // audio.
        // see documentation at:
        // https://mf4.xiph.org/jenkins/view/opus/job/opus/ws/doc/html/group__opus__decoder.html#ga7d1111f64c36027ddcb81799df9b3fc9
        std::vector<opus_int16> Decode(
                const std::vector<std::vector<unsigned char>>& packets, int frame_size,
                bool decode_fec);

        int valid() const { return valid_; }

        // Takes an encoded packet and decodes it. Returns the decoded audio
        // see documentation at:
        // https://mf4.xiph.org/jenkins/view/opus/job/opus/ws/doc/html/group__opus__decoder.html#ga7d1111f64c36027ddcb81799df9b3fc9
        std::vector<opus_int16> Decode(const std::vector<unsigned char>& packet,
                                       int frame_size, bool decode_fec);

        // Generates a dummy frame by passing nullptr to the underlying opus decode.
        std::vector<opus_int16> DecodeDummy(int frame_size);

    private:
        int num_channels_{};
        bool valid_{};
        internal::opus_uptr<OpusDecoder> decoder_;
    };

}  // namespace opus

#endif