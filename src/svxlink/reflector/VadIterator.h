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

    timestamp_t &operator=(const timestamp_t &a);

    bool operator==(const timestamp_t &a) const;

    std::string c_str();

private:
    std::string format(const char *fmt, ...);
};

class VadIterator {
public:
    /***
    From https://github.com/SYSTRAN/faster-whisper/blob/1eb9a8004c509a4af2960955374520c35b7b793a/faster_whisper/vad.py#L15
    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      window_size_samples: Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate.
        Values other than these may affect model performance!!
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    window_size_samples: int = 1024
    speech_pad_ms: int = 400
      */

    VadIterator(const std::wstring ModelPath,
                int Sample_rate = 16000, int windows_frame_size = 64,
                float Threshold = 0.3, int min_silence_duration_ms = 50,
                int speech_pad_ms = 30, int min_speech_duration_ms = 1000,
                float max_speech_duration_s = std::numeric_limits<float>::infinity());

    void process(const std::vector<float> &input_wav);

    void process(const std::vector<float> &input_wav, std::vector<float> &output_wav);

    const std::vector<timestamp_t> get_speech_timestamps() const;

    void drop_chunks(const std::vector<float> &input_wav, std::vector<float> &output_wav);

    void collect_chunks(const std::vector<float> &input_wav, std::vector<float> &output_wav);

    bool isVoicePresent() const;

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