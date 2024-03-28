#include "VadIterator.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <codecvt>
#include <locale>
#include <algorithm>
#include <chrono>
#include <iomanip>

// Implementation of timestamp_t class
timestamp_t::timestamp_t(int start, int end) : start(start), end(end) {}

timestamp_t &timestamp_t::operator=(const timestamp_t &a) {
    start = a.start;
    end = a.end;
    return *this;
}

bool timestamp_t::operator==(const timestamp_t &a) const {
    return start == a.start && end == a.end;
}

std::string timestamp_t::c_str() {
    return format("{start:%08d,end:%08d}", start, end);
}

std::string timestamp_t::format(const char *fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    const auto r = std::vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);

    if (r < 0) {
        return {};
    }

    const size_t len = r;
    if (len < sizeof buf) {
        return {buf, len};
    }

#if __cplusplus >= 201703L
    std::string s(len, '\0');
    va_start(args, fmt);
    std::vsnprintf(s.data(), len + 1, fmt, args);
    va_end(args);
    return s;
#else
    auto vbuf = std::unique_ptr<char[]>(new char[len + 1]);
    va_start(args, fmt);
    std::vsnprintf(vbuf.get(), len + 1, fmt, args);
    va_end(args);
    return {vbuf.get(), len};
#endif
}

// Implementation of VadIterator class
VadIterator::VadIterator(const std::wstring ModelPath, int Sample_rate, int64_t window_size_samples, float Threshold,
                         int min_silence_duration_ms, int speech_pad_ms, int min_speech_duration_ms,
                         float max_speech_duration_s)
        : window_size_samples(window_size_samples), // Initialize sample_rate with Sample_rate
          sample_rate(Sample_rate), // Initialize window_size_samples with window_size_samples
          sr_per_ms(Sample_rate / 1000), // Initialize threshold with Threshold
          threshold(Threshold), // Calculate sr_per_ms based on Sample_rate
          min_silence_samples(sr_per_ms * min_silence_duration_ms), // Calculate min_speech_samples
          min_silence_samples_at_max_speech(sr_per_ms * 98), // Calculate speech_pad_samples
          min_speech_samples(sr_per_ms * min_speech_duration_ms), // Calculate min_silence_samples
          max_speech_samples(Sample_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples), // Initialize min_silence_samples_at_max_speech
          speech_pad_samples(sr_per_ms * speech_pad_ms) // Calculate max_speech_samples
{
    init_onnx_model(ModelPath);

    // Since input, _h, _c, and sr are likely std::vector or similar, their sizes can't be set in the initializer list,
    // but you can resize them immediately after in the constructor body.
    input.resize(window_size_samples);
    input_node_dims[0] = 1;
    input_node_dims[1] = window_size_samples;

    _h.resize(size_hc); // Assuming size_hc is already set correctly before this constructor is called
    _c.resize(size_hc);
    sr.resize(1);
    sr[0] = sample_rate;
}

void VadIterator::init_engine_threads(int inter_threads, int intra_threads) {
    session_options.SetIntraOpNumThreads(intra_threads);
    session_options.SetInterOpNumThreads(inter_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

void VadIterator::init_onnx_model(const std::wstring &model_path) {
    init_engine_threads(1, 1);

    // Convert std::wstring to std::string (assuming UTF-8 encoding)
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::string narrowModelPath = conv.to_bytes(model_path);

    // Use the narrow string for the model path
    session = std::make_shared<Ort::Session>(env, narrowModelPath.c_str(), session_options);
}

void VadIterator::reset_states() {
    std::memset(_h.data(), 0.0f, _h.size() * sizeof(float));
    std::memset(_c.data(), 0.0f, _c.size() * sizeof(float));
    triggered = false;
    temp_end = 0;
    current_sample = 0;
    prev_end = next_start = 0;
    speeches.clear();
    current_speech = timestamp_t();
}

void VadIterator::predict(const std::vector<float> &data) {
    // Infer
    // Create ort tensors
    input.assign(data.begin(), data.end());
    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
    Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);
    Ort::Value h_ort = Ort::Value::CreateTensor<float>(
            memory_info, _h.data(), _h.size(), hc_node_dims, 3);
    Ort::Value c_ort = Ort::Value::CreateTensor<float>(
            memory_info, _c.data(), _c.size(), hc_node_dims, 3);

    // Clear and add inputs
    ort_inputs.clear();
    ort_inputs.emplace_back(std::move(input_ort));
    ort_inputs.emplace_back(std::move(sr_ort));
    ort_inputs.emplace_back(std::move(h_ort));
    ort_inputs.emplace_back(std::move(c_ort));

    // Infer
    ort_outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

    // Output probability & update h,c recursively
    float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
    float *hn = ort_outputs[1].GetTensorMutableData<float>();
    std::memcpy(_h.data(), hn, size_hc * sizeof(float));
    float *cn = ort_outputs[2].GetTensorMutableData<float>();
    std::memcpy(_c.data(), cn, size_hc * sizeof(float));

    float lastMaxAmplitude = *std::max_element(data.begin(), data.end());
    // debug the speech probability in percentage
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto localTime = std::localtime(&now_c);

    // print only when speech_prob >= threshold
    if (speech_prob >= threshold)
    {
        std::cout << "Voice detected at " << std::put_time(localTime, "%H:%M:%S") << " - probability: " << speech_prob * 100
                  << "% lastMaxAmplitude: " << lastMaxAmplitude << std::endl;
    }

    // Push forward sample index
    current_sample += window_size_samples;

    // Reset temp_end when > threshold
    if ((speech_prob >= threshold)) {
        voiceDetected = true;
#ifdef __DEBUG_SPEECH_PROB___
        float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
            printf("{    start: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample- window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        if (temp_end != 0) {
            temp_end = 0;
            if (next_start < prev_end)
                next_start = current_sample - window_size_samples;
        }
        if (triggered == false) {
            triggered = true;

            current_speech.start = current_sample - window_size_samples;
        }
        return;
    }

    if (
            (triggered == true)
            && ((current_sample - current_speech.start) > max_speech_samples)
            ) {
        if (prev_end > 0) {
            current_speech.end = prev_end;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();

            // previously reached silence(< neg_thres) and is still not speech(< thres)
            if (next_start < prev_end)
                triggered = false;
            else {
                current_speech.start = next_start;
            }
            prev_end = 0;
            next_start = 0;
            temp_end = 0;

        } else {
            current_speech.end = current_sample;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
        }
        return;

    }
    if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold)) {
        if (triggered) {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{ speeking: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        } else {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{  silence: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        }
        return;
    }


    // 4) End
    if ((speech_prob < (threshold - 0.15))) {
#ifdef __DEBUG_SPEECH_PROB___
        float speech = current_sample - window_size_samples - speech_pad_samples; // minus window_size_samples to get precise start time point.
            printf("{      end: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        if (triggered == true) {
            if (temp_end == 0) {
                temp_end = current_sample;
            }
            if (current_sample - temp_end > min_silence_samples_at_max_speech)
                prev_end = temp_end;
            // a. silence < min_slience_samples, continue speaking
            if ((current_sample - temp_end) < min_silence_samples) {

            }
                // b. silence >= min_slience_samples, end speaking
            else {
                current_speech.end = temp_end;
                if (current_speech.end - current_speech.start > min_speech_samples) {
                    speeches.push_back(current_speech);
                    current_speech = timestamp_t();
                    prev_end = 0;
                    next_start = 0;
                    temp_end = 0;
                    triggered = false;
                }
            }
        } else {
            // may first windows see end state.
        }
        return;
    }
}

void VadIterator::process(const std::vector<float> &input_wav) {
    reset_states();
    voiceDetected = false;

    audio_length_samples = input_wav.size();

    for (int j = 0; j < audio_length_samples; j += window_size_samples) {
        if (j + window_size_samples > audio_length_samples)
            break;
        std::vector<float> r{&input_wav[0] + j, &input_wav[0] + j + window_size_samples};
        predict(r);
    }

    if (current_speech.start >= 0) {
        current_speech.end = audio_length_samples;
        speeches.push_back(current_speech);
        current_speech = timestamp_t();
        prev_end = 0;
        next_start = 0;
        temp_end = 0;
        triggered = false;
    }
}

void VadIterator::process(const std::vector<float> &input_wav, std::vector<float> &output_wav) {
    process(input_wav);
    collect_chunks(input_wav, output_wav);
}


void VadIterator::collect_chunks(const std::vector<float> &input_wav, std::vector<float> &output_wav) {
    output_wav.clear();
    for (const auto &speech: speeches) {
        std::vector<float> slice(&input_wav[speech.start], &input_wav[speech.end]);
        output_wav.insert(output_wav.end(), slice.begin(), slice.end());
    }
}

const std::vector<timestamp_t> VadIterator::get_speech_timestamps() const {
    return speeches;
}

void VadIterator::drop_chunks(const std::vector<float> &input_wav, std::vector<float> &output_wav) {
    output_wav.clear();
    int current_start = 0;
    for (const auto &speech: speeches) {
        std::vector<float> slice(&input_wav[current_start], &input_wav[speech.start]);
        output_wav.insert(output_wav.end(), slice.begin(), slice.end());
        current_start = speech.end;
    }

    std::vector<float> slice(&input_wav[current_start], &input_wav[input_wav.size()]);
    output_wav.insert(output_wav.end(), slice.begin(), slice.end());
}
