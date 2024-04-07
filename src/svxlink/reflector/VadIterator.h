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