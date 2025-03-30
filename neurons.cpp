#include "neurons.hpp"
#include <random>
#include <cassert>

// Conditional neuron configuration
#if USE_CUDA
constexpr size_t HiddenSize = 90'000'000'000; // 90 Billion neurons with CUDA
#else
constexpr size_t HiddenSize = 10'000;         // 10,000 neurons without CUDA
#endif

constexpr size_t InputSize = 784;   // 28x28 MNIST
constexpr size_t OutputSize = 10;   // Digits 0-9

// Constructor
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
NeuralNetwork<InputSize, HiddenSize, OutputSize>::NeuralNetwork() {
    inputLayer.resize(InputSize);
    hiddenLayer.resize(HiddenSize);
    outputLayer.resize(OutputSize);

    weights1.resize(InputSize * HiddenSize);
    weights2.resize(HiddenSize * OutputSize);
    bias1.resize(HiddenSize, 0.0f);
    bias2.resize(OutputSize, 0.0f);

    initializeWeights();

#if USE_CUDA
    // Allocate GPU memory for CUDA
    cudaMalloc(&d_input, InputSize * sizeof(float));
    cudaMalloc(&d_hidden, HiddenSize * sizeof(float));
    cudaMalloc(&d_output, OutputSize * sizeof(float));

    cudaMalloc(&d_weights1, InputSize * HiddenSize * sizeof(float));
    cudaMalloc(&d_weights2, HiddenSize * OutputSize * sizeof(float));
    cudaMalloc(&d_bias1, HiddenSize * sizeof(float));
    cudaMalloc(&d_bias2, OutputSize * sizeof(float));
#endif
}

// Destructor
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
NeuralNetwork<InputSize, HiddenSize, OutputSize>::~NeuralNetwork() {
#if USE_CUDA
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_weights1);
    cudaFree(d_weights2);
    cudaFree(d_bias1);
    cudaFree(d_bias2);
#endif
}

// Xavier initialization
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
void NeuralNetwork<InputSize, HiddenSize, OutputSize>::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit1 = std::sqrt(6.0f / (InputSize + HiddenSize));
    float limit2 = std::sqrt(6.0f / (HiddenSize + OutputSize));

    std::uniform_real_distribution<float> dist1(-limit1, limit1);
    std::uniform_real_distribution<float> dist2(-limit2, limit2);

    for (auto& w : weights1) w = dist1(gen);
    for (auto& w : weights2) w = dist2(gen);
}

// ReLU Activation
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
void NeuralNetwork<InputSize, HiddenSize, OutputSize>::reluActivation(std::vector<float>& data) {
    for (float& x : data) {
        x = std::max(0.0f, x);
    }
}

// Softmax Activation
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
void NeuralNetwork<InputSize, HiddenSize, OutputSize>::softmaxActivation(std::vector<float>& data) {
    float maxVal = *std::max_element(data.begin(), data.end());
    float sum = 0.0f;
    for (float& x : data) {
        x = std::exp(x - maxVal);
        sum += x;
    }
    for (float& x : data) {
        x /= sum;
    }
}

// Forward Pass
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
void NeuralNetwork<InputSize, HiddenSize, OutputSize>::forward(const std::vector<float>& input) {
    assert(input.size() == InputSize);

    // Input to Hidden Layer
    for (size_t i = 0; i < HiddenSize; i++) {
        float sum = bias1[i];
        for (size_t j = 0; j < InputSize; j++) {
            sum += input[j] * weights1[j * HiddenSize + i];
        }
        hiddenLayer[i] = sum;
    }
    reluActivation(hiddenLayer);

    // Hidden to Output Layer
    for (size_t i = 0; i < OutputSize; i++) {
        float sum = bias2[i];
        for (size_t j = 0; j < HiddenSize; j++) {
            sum += hiddenLayer[j] * weights2[j * OutputSize + i];
        }
        outputLayer[i] = sum;
    }
    softmaxActivation(outputLayer);
}

// Print Output
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
void NeuralNetwork<InputSize, HiddenSize, OutputSize>::printOutput() const {
    std::cout << "Output: ";
    for (float val : outputLayer) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// Explicit template instantiation
template class NeuralNetwork<InputSize, HiddenSize, OutputSize>;

template<>
void NeuralNetwork<InputSize, HiddenSize, OutputSize>::backward(const std::vector<float>& target, float learningRate) {
    std::vector<float> outputError(OutputSize);
    std::vector<float> hiddenError(HiddenSize);

    // Calculate Output Error (dL/dOutput)
    for (size_t i = 0; i < OutputSize; i++) {
        outputError[i] = outputLayer[i] - target[i];
    }

    // Calculate Hidden Error using Backpropagation
    for (size_t i = 0; i < HiddenSize; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < OutputSize; j++) {
            sum += outputError[j] * weights2[i * OutputSize + j];
        }
        hiddenError[i] = (hiddenLayer[i] > 0 ? 1.0f : 0.0f) * sum; // ReLU Derivative
    }

    // Update Weights2 and Bias2 (Output Layer)
    for (size_t i = 0; i < OutputSize; i++) {
        for (size_t j = 0; j < HiddenSize; j++) {
            weights2[j * OutputSize + i] -= learningRate * outputError[i] * hiddenLayer[j];
        }
        bias2[i] -= learningRate * outputError[i];
    }

    // Update Weights1 and Bias1 (Hidden Layer)
    for (size_t i = 0; i < HiddenSize; i++) {
        for (size_t j = 0; j < InputSize; j++) {
            weights1[j * HiddenSize + i] -= learningRate * hiddenError[i] * inputLayer[j];
        }
        bias1[i] -= learningRate * hiddenError[i];
    }
}
