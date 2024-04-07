#include "VadIterator.h"
#include <cstring>
#include <codecvt>
#include <locale>
#include <algorithm>
#include <chrono>
#include <iomanip>

// Implementation of VadIterator class
VadIterator::VadIterator(const std::wstring& ModelPath, int Sample_rate, int64_t window_size_samples, float Threshold)
        : window_size_samples(window_size_samples),
          sample_rate(Sample_rate),
          threshold(Threshold)
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
    voiceDetected = false;
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
    auto *hn = ort_outputs[1].GetTensorMutableData<float>();
    std::memcpy(_h.data(), hn, size_hc * sizeof(float));
    auto *cn = ort_outputs[2].GetTensorMutableData<float>();
    std::memcpy(_c.data(), cn, size_hc * sizeof(float));

    float lastMaxAmplitude = *std::max_element(data.begin(), data.end());
    // debug the speech probability in percentage
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto localTime = std::localtime(&now_c);

    // print only when speech_prob >= threshold
    if (speech_prob >= threshold)
    {
        voiceDetected = true;
        std::cout << "Voice detected at " << std::put_time(localTime, "%H:%M:%S") << " - probability: " << speech_prob * 100
                  << "% lastMaxAmplitude: " << lastMaxAmplitude << std::endl;
        return;
    }
}

void VadIterator::process(const std::vector<float> &input_wav) {
    reset_states();

    audio_length_samples = input_wav.size();

    for (int64_t j = 0; j < audio_length_samples; j += window_size_samples) {
        if (j + window_size_samples > audio_length_samples)
            break;
        std::vector<float> r{&input_wav[0] + j, &input_wav[0] + j + window_size_samples};
        predict(r);
    }

}
