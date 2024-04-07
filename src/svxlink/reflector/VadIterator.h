/*
 * VadIterator.h
 *
 * This file is part of the SvxLink project, which is licensed under the GNU General Public License v2.
 * Portions of this file are inspired by or derived from code found in https://github.com/snakers4/silero-vad/,
 * a project licensed under the MIT License. The specific source file did not contain a separate license header,
 * but the project as a whole is licensed under the MIT License, as detailed in the project's repository.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (included in the project's repository)
 * shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Additional modifications and contributions to this file are licensed under the GNU General Public License v2,
 * and are copyrighted (C) 2024 Silviu Stroe - www.brainic.io
 */

#ifndef VADITERATOR_H
#define VADITERATOR_H

#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>
#include "onnxruntime_cxx_api.h"

class VadIterator {
public:

    explicit VadIterator(const std::wstring& ModelPath,
                int Sample_rate = 16000, int64_t window_size_samples = 1536,
                float Threshold = 0.3);

    void process(const std::vector<float> &input_wav);

    bool isVoicePresent() const { return voiceDetected; }

private:
    void init_engine_threads(int inter_threads, int intra_threads);

    void init_onnx_model(const std::wstring &model_path);

    void reset_states();

    void predict(const std::vector<float> &data);

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    int64_t window_size_samples;
    int sample_rate;
    float threshold;
    unsigned long audio_length_samples{};

    std::vector<Ort::Value> ort_inputs;

    std::vector<const char *> input_node_names = {"input", "sr", "h", "c"};
    std::vector<float> input;
    std::vector<int64_t> sr;
    unsigned int size_hc = 2 * 1 * 64;
    std::vector<float> _h;
    std::vector<float> _c;

    int64_t input_node_dims[2] = {};
    const int64_t sr_node_dims[1] = {1};
    const int64_t hc_node_dims[3] = {2, 1, 64};

    std::vector<Ort::Value> ort_outputs;
    std::vector<const char *> output_node_names = {"output", "hn", "cn"};

    std::vector<float> accumulationBuffer;
    bool voiceDetected = false;
};

#endif // VADITERATOR_H