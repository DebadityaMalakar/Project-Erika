#include "loader.hpp"
#include "neurons.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Configuration
constexpr size_t InputSize = 784;   // 28x28 MNIST
#if USE_CUDA
constexpr size_t HiddenSize = 90'000'000'000;  // 90 Billion neurons
#else
constexpr size_t HiddenSize = 10'000;          // 10,000 neurons
#endif
constexpr size_t OutputSize = 10;   // Digits 0-9
constexpr float LearningRate = 0.001;
constexpr int Epochs = 10;

void oneHotEncode(int label, std::vector<float>& target) {
    target.assign(OutputSize, 0.0f);
    target[label] = 1.0f;
}

int getPrediction(const std::vector<float>& output) {
    return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}

void train(NeuralNetwork<InputSize, HiddenSize, OutputSize>& nn, const std::vector<DataPoint>& trainData) {
    std::cout << "Training Started...\n";

    for (int epoch = 0; epoch < Epochs; epoch++) {
        float totalLoss = 0.0;
        int correctCount = 0;

        for (const auto& dataPoint : trainData) {
            nn.forward(dataPoint.pixels);

            std::vector<float> target(OutputSize);
            oneHotEncode(dataPoint.label, target);

            int prediction = getPrediction(nn.getOutputLayer());
            if (prediction == dataPoint.label) correctCount++;

            nn.backward(target, LearningRate);

            // Simple Loss Calculation (Cross-Entropy)
            float loss = 0.0;
            const std::vector<float>& outputLayer = nn.getOutputLayer();
            for (size_t j = 0; j < OutputSize; j++) {
                loss -= target[j] * std::log(outputLayer[j] + 1e-7f);
            }
            totalLoss += loss;
        }

        float accuracy = (float)correctCount / trainData.size();
        std::cout << "Epoch " << epoch + 1 << " | Loss: " << totalLoss / trainData.size() << " | Accuracy: " << accuracy * 100.0f << "%\n";
    }
}

void test(NeuralNetwork<InputSize, HiddenSize, OutputSize>& nn, const std::vector<DataPoint>& testData) {
    std::cout << "Testing Started...\n";

    int correctCount = 0;
    for (const auto& dataPoint : testData) {
        nn.forward(dataPoint.pixels);
        int prediction = getPrediction(nn.getOutputLayer());
        if (prediction == dataPoint.label) correctCount++;
    }

    float accuracy = (float)correctCount / testData.size();
    std::cout << "Test Accuracy: " << accuracy * 100.0f << "%\n";
}

int main() {
    std::cout << "Loading Data...\n";
    std::vector<DataPoint> trainData = loadCSV("mnist_train.csv");
    std::vector<DataPoint> testData = loadCSV("mnist_test.csv");

    std::cout << "Initializing Neural Network...\n";
    NeuralNetwork<InputSize, HiddenSize, OutputSize> nn;

    train(nn, trainData);
    test(nn, testData);

    return 0;
}
