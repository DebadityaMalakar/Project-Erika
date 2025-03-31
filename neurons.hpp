#ifndef NEURONS_HPP
#define NEURONS_HPP

#include <vector>
#include <iostream>
#include <cmath>
#include <random>

// Check if CUDA is available
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define USE_CUDA 1
#else
#define USE_CUDA 0
#endif

template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();

    void forward(const std::vector<float>& input);
    void backward(const std::vector<float>& target, float learningRate);
    void printOutput() const;

    std::vector<float> getOutputLayer() const {
        return outputLayer;
    }

private:
    std::vector<float> inputLayer;
    std::vector<float> hiddenLayer;
    std::vector<float> outputLayer;

    std::vector<float> weights1;
    std::vector<float> weights2;
    std::vector<float> bias1;
    std::vector<float> bias2;

    void initializeWeights();

    // Inline activation functions
    void reluActivation(std::vector<float>& data) {
        for (float& x : data) x = std::max(0.0f, x);
    }

    void softmaxActivation(std::vector<float>& data) {
        float maxVal = *std::max_element(data.begin(), data.end());
        float sum = 0.0f;
        for (float& x : data) {
            x = std::exp(x - maxVal);
            sum += x;
        }
        for (float& x : data) x /= sum;
    }

#if USE_CUDA
    float *d_input, *d_hidden, *d_output;
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    void gpuMatrixMultiply(float* A, float* B, float* C, size_t M, size_t N, size_t K);
#endif
};

#endif // NEURONS_HPP
