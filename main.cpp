#include "loader.hpp"
#include "neurons.hpp"
#include "model.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <limits>

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

void test(NeuralNetwork<InputSize, HiddenSize, OutputSize>& nn, const std::vector<DataPoint>& testData);

void oneHotEncode(int label, std::vector<float>& target) {
    target.assign(OutputSize, 0.0f);
    target[label] = 1.0f;
}

int getPrediction(const std::vector<float>& output) {
    return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}

void train(NeuralNetwork<InputSize, HiddenSize, OutputSize>& nn, 
           const std::vector<DataPoint>& trainData,
           const std::vector<DataPoint>& testData) {
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

    // Test after training
    test(nn, testData);
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

void displayMenu() {
    clearScreen();
    std::cout << "Project Erika - Main Menu\n\n";
    std::cout << "1. Load and test a model\n";
    std::cout << "2. Train a new model\n";
    std::cout << "3. Exit\n";
    std::cout << "\nEnter your choice (1-3): ";
}

void loadAndTestModel() {
    clearScreen();
    std::cout << "Load and Test Model\n\n";
    
    NeuralNetwork<InputSize, HiddenSize, OutputSize> nn;
    
    std::string modelFilename;
    std::cout << "Enter model filename to load (e.g., model.erk): ";
    std::cin >> modelFilename;
    
    if (!ModelManager<InputSize, HiddenSize, OutputSize>::loadModel(nn, modelFilename)) {
        std::cerr << "Failed to load model. Press Enter to return to menu...";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
        return;
    }
    
    std::string testFilename;
    std::cout << "Enter test CSV filename (e.g., mnist_test.csv): ";
    std::cin >> testFilename;
    
    std::vector<DataPoint> testData = loadCSV(testFilename);
    if (testData.empty()) {
        std::cerr << "Failed to load test data or file is empty. Press Enter to return to menu...";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
        return;
    }
    
    test(nn, testData);
    
    std::cout << "\nPress Enter to return to menu...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
}

void trainNewModel() {
    clearScreen();
    std::cout << "Train New Model\n\n";
    
    std::string trainFilename, testFilename;
    std::cout << "Enter training CSV filename (e.g., mnist_train.csv): ";
    std::cin >> trainFilename;
    std::cout << "Enter test CSV filename (e.g., mnist_test.csv): ";
    std::cin >> testFilename;
    
    std::vector<DataPoint> trainData = loadCSV(trainFilename);
    std::vector<DataPoint> testData = loadCSV(testFilename);
    
    if (trainData.empty() || testData.empty()) {
        std::cerr << "Failed to load data or files are empty. Press Enter to return to menu...";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
        return;
    }
    
    NeuralNetwork<InputSize, HiddenSize, OutputSize> nn;
    train(nn, trainData, testData);
    
    char saveChoice;
    std::cout << "\nDo you want to save the trained model? (y/n): ";
    std::cin >> saveChoice;
    
    if (saveChoice == 'y' || saveChoice == 'Y') {
        std::string modelFilename;
        std::cout << "Enter filename to save the model (e.g., model.erk): ";
        std::cin >> modelFilename;
        if (!ModelManager<InputSize, HiddenSize, OutputSize>::saveModel(nn, modelFilename)) {
            std::cerr << "Failed to save model.\n";
        } else {
            std::cout << "Model saved successfully!\n";
        }
    }
    
    std::cout << "\nPress Enter to return to menu...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
}

int main() {
    while (true) {
        displayMenu();
        
        int choice;
        std::cin >> choice;
        
        switch (choice) {
            case 1:
                loadAndTestModel();
                break;
            case 2:
                trainNewModel();
                break;
            case 3:
                std::cout << "Exiting Project Erika. Goodbye!\n";
                return 0;
            default:
                std::cout << "Invalid choice. Please try again.\n";
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cin.get();
                break;
        }
    }

    return 0;
}