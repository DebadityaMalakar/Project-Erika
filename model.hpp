#ifndef MODEL_HPP
#define MODEL_HPP

#include "neurons.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
class ModelManager {
public:
    static bool saveModel(const NeuralNetwork<InputSize, HiddenSize, OutputSize>& nn, const std::string& filename);
    static bool loadModel(NeuralNetwork<InputSize, HiddenSize, OutputSize>& nn, const std::string& filename);
};

#endif // MODEL_HPP
