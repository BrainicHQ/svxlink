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
#include "wav.h"

class timestamp_t {
public:
    int start;
    int end;

    timestamp_t(int start = -1, int end = -1);
    timestamp_t& operator=(const timestamp_t& a);
    bool operator==(const timestamp_t& a) const;
    std::string c_str();

private:
    std::string format(const char* fmt, ...);
};

class VadIterator {
public:
    VadIterator(const std::wstring ModelPath,
                int Sample_rate = 16000, int windows_frame_size = 64,
                float Threshold = 0.5, int min_silence_duration_ms = 2000,
                int speech_pad_ms = 400, int min_speech_duration_ms = 250,
                float max_speech_duration_s = std::numeric_limits<float>::infinity());

    void process(const std::vector<float>& input_wav);
    void process(const std::vector<float>& input_wav, std::vector<float>& output_wav);
    const std::vector<timestamp_t> get_speech_timestamps() const;
    void drop_chunks(const std::vector<float>& input_wav, std::vector<float>& output_wav);
    void collect_chunks(const std::vector<float>& input_wav, std::vector<float>& output_wav);

    bool isVoicePresent() const;

private:
    void init_engine_threads(int inter_threads, int intra_threads);
    void init_onnx_model(const std::wstring& model_path);
    void reset_states();
    void predict(const std::vector<float> &data);

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    int64_t window_size_samples;
    int sample_rate;
    int sr_per_ms;
    float threshold;
    int min_silence_samples;
    int min_silence_samples_at_max_speech;
    int min_speech_samples;
    float max_speech_samples;
    int speech_pad_samples;
    int64_t audio_length_samples;

    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;

    int prev_end;
    int next_start = 0;

    std::vector<timestamp_t> speeches;
    timestamp_t current_speech;

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
};

#endif // VADITERATOR_H