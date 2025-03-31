#include "neurons.hpp"
#include <cassert>
#include <omp.h>

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

    #pragma omp parallel for
    for (size_t i = 0; i < InputSize * HiddenSize; i++) {
        weights1[i] = dist1(gen);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < HiddenSize * OutputSize; i++) {
        weights2[i] = dist2(gen);
    }
}

// Forward Pass
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
void NeuralNetwork<InputSize, HiddenSize, OutputSize>::forward(const std::vector<float>& input) {
    assert(input.size() == InputSize);

    // Compute hidden layer
    #pragma omp parallel for
    for (size_t i = 0; i < HiddenSize; i++) {
        float sum = bias1[i];
        for (size_t j = 0; j < InputSize; j++) {
            sum += input[j] * weights1[j * HiddenSize + i];
        }
        hiddenLayer[i] = sum;
    }
    reluActivation(hiddenLayer);

    // Compute output layer
    #pragma omp parallel for
    for (size_t i = 0; i < OutputSize; i++) {
        float sum = bias2[i];
        for (size_t j = 0; j < HiddenSize; j++) {
            sum += hiddenLayer[j] * weights2[j * OutputSize + i];
        }
        outputLayer[i] = sum;
    }
    softmaxActivation(outputLayer);
}

// Backpropagation
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
void NeuralNetwork<InputSize, HiddenSize, OutputSize>::backward(const std::vector<float>& target, float learningRate) {
    std::vector<float> outputError(OutputSize);
    std::vector<float> hiddenError(HiddenSize);

    // Calculate output error
    #pragma omp parallel for
    for (size_t i = 0; i < OutputSize; i++) {
        outputError[i] = outputLayer[i] - target[i];
    }

    // Calculate hidden layer error
    #pragma omp parallel for
    for (size_t i = 0; i < HiddenSize; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < OutputSize; j++) {
            sum += outputError[j] * weights2[i * OutputSize + j];
        }
        hiddenError[i] = (hiddenLayer[i] > 0 ? 1.0f : 0.0f) * sum;
    }

    // Update weights2 and biases2
    #pragma omp parallel for
    for (size_t i = 0; i < OutputSize; i++) {
        for (size_t j = 0; j < HiddenSize; j++) {
            weights2[j * OutputSize + i] -= learningRate * outputError[i] * hiddenLayer[j];
        }
        bias2[i] -= learningRate * outputError[i];
    }

    // Update weights1 and biases1
    #pragma omp parallel for
    for (size_t i = 0; i < HiddenSize; i++) {
        for (size_t j = 0; j < InputSize; j++) {
            weights1[j * HiddenSize + i] -= learningRate * hiddenError[i] * inputLayer[j];
        }
        bias1[i] -= learningRate * hiddenError[i];
    }
}

// Print Output
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
void NeuralNetwork<InputSize, HiddenSize, OutputSize>::printOutput() const {
    std::cout << "Output: ";
    for (float val : outputLayer) std::cout << val << " ";
    std::cout << std::endl;
}

// Explicit Instantiation
template class NeuralNetwork<784, 1024, 10>;
