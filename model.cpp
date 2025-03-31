#include "model.hpp"

template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
bool ModelManager<InputSize, HiddenSize, OutputSize>::saveModel(const NeuralNetwork<InputSize, HiddenSize, OutputSize>& nn, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for saving: " << filename << std::endl;
        return false;
    }

    std::cout << "Saving model to " << filename << "...\n";

    // Save weights and biases
    const auto& weights1 = nn.getWeights1();
    const auto& weights2 = nn.getWeights2();
    const auto& bias1 = nn.getBias1();
    const auto& bias2 = nn.getBias2();

    file.write(reinterpret_cast<const char*>(weights1.data()), weights1.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(weights2.data()), weights2.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bias1.data()), bias1.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bias2.data()), bias2.size() * sizeof(float));
    

    file.close();
    std::cout << "Model saved successfully!\n";
    return true;
}

template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
bool ModelManager<InputSize, HiddenSize, OutputSize>::loadModel(NeuralNetwork<InputSize, HiddenSize, OutputSize>& nn, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for loading: " << filename << std::endl;
        return false;
    }

    std::cout << "Loading model from " << filename << "...\n";

    // Create temporary buffers to hold the data
    std::vector<float> weights1_buf(InputSize * HiddenSize);
    std::vector<float> weights2_buf(HiddenSize * OutputSize);
    std::vector<float> bias1_buf(HiddenSize);
    std::vector<float> bias2_buf(OutputSize);

    // Read into temporary buffers
    file.read(reinterpret_cast<char*>(weights1_buf.data()), weights1_buf.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(weights2_buf.data()), weights2_buf.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(bias1_buf.data()), bias1_buf.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(bias2_buf.data()), bias2_buf.size() * sizeof(float));

    file.close();

    // Update the neural network with the loaded values
    nn.setWeights1(weights1_buf);
    nn.setWeights2(weights2_buf);
    nn.setBias1(bias1_buf);
    nn.setBias2(bias2_buf);

    std::cout << "Model loaded successfully!\n";
    return true;
}

template class ModelManager<784, 1024, 10>;
template class ModelManager<784, 10000, 10>;