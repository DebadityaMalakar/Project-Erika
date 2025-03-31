#include "loader.hpp"
#include "neurons.hpp"
#include "model.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

// Configuration
constexpr size_t InputSize = 784;   // 28x28 MNIST
#if USE_CUDA
constexpr size_t HiddenSize = 10'000; 
#else
constexpr size_t HiddenSize = 1024;
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

    auto trainingStart = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < Epochs; epoch++) {
        auto epochStart = std::chrono::high_resolution_clock::now();

        float totalLoss = 0.0;
        int correctCount = 0;

        for (const auto& dataPoint : trainData) {
            nn.forward(dataPoint.pixels);

            std::vector<float> target(OutputSize);
            oneHotEncode(dataPoint.label, target);

            int prediction = getPrediction(nn.getOutputLayer());
            if (prediction == dataPoint.label) correctCount++;

            nn.backward(target, LearningRate);

            float loss = 0.0;
            const std::vector<float>& outputLayer = nn.getOutputLayer();
            for (size_t j = 0; j < OutputSize; j++) {
                loss -= target[j] * std::log(outputLayer[j] + 1e-7f);
            }
            totalLoss += loss;
        }

        auto epochEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epochDuration = epochEnd - epochStart;

        float accuracy = (float)correctCount / trainData.size();
        std::cout << "Epoch " << epoch + 1 << " / " << Epochs 
                  << " completed | Loss: " << totalLoss / trainData.size() 
                  << " | Accuracy: " << accuracy * 100.0f << "%"
                  << " | Time: " << epochDuration.count() << "s\n";
    }

    auto trainingEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> trainingDuration = trainingEnd - trainingStart;

    std::cout << "Total Training Time: " << trainingDuration.count() << "s\n";
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

void clearScreen() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

int main() {
    clearScreen();
    std::cout << "Project Erika has started\n\n";

    // Load Data
    std::cout << "Loading Data...\n";
    std::vector<DataPoint> trainData = loadCSV("mnist_train.csv");
    std::vector<DataPoint> testData = loadCSV("mnist_test.csv");

    std::cout << "Initializing Neural Network...\n";
    NeuralNetwork<InputSize, HiddenSize, OutputSize> nn;

    // Ask to load model
    char loadChoice;
    std::cout << "Do you want to load a pre-trained model? (y/n): ";
    std::cin >> loadChoice;

    if (loadChoice == 'y' || loadChoice == 'Y') {
        std::string filename;
        std::cout << "Enter model filename (e.g., model.erk): ";
        std::cin >> filename;
        if (!ModelManager<InputSize, HiddenSize, OutputSize>::loadModel(nn, filename)) {
            std::cerr << "Failed to load model. Starting with a fresh model.\n";
        }
    } else {
        std::cout << "Starting with a new model.\n";
    }

    // Train and Test
    train(nn, trainData);
    test(nn, testData);

    // Ask to save the model
    char saveChoice;
    std::cout << "Do you want to save the trained model? (y/n): ";
    std::cin >> saveChoice;

    if (saveChoice == 'y' || saveChoice == 'Y') {
        std::string filename;
        std::cout << "Enter filename to save the model (e.g., model.erk): ";
        std::cin >> filename;
        if (!ModelManager<InputSize, HiddenSize, OutputSize>::saveModel(nn, filename)) {
            std::cerr << "Failed to save model.\n";
        } else {
            std::cout << "Model saved successfully!\n";
        }
    }

    return 0;
}
