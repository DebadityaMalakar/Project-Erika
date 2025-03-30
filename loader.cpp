#include "loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

// Function to load MNIST data from CSV
std::vector<DataPoint> loadCSV(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        DataPoint point;

        // Read label (first column)
        std::getline(ss, value, ',');
        point.label = std::stoi(value);

        // Read 784 pixel values
        while (std::getline(ss, value, ',')) {
            point.pixels.push_back(std::stof(value));
        }

        // Validate data
        if (point.pixels.size() != 784) {
            std::cerr << "Error: Data format mismatch. Expected 784 pixels, got " 
                      << point.pixels.size() << std::endl;
            continue;
        }

        data.push_back(point);
    }

    std::cout << "Successfully loaded " << data.size() << " data points from " << filename << std::endl;
    return data;
}
