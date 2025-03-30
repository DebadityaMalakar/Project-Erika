#ifndef LOADER_HPP
#define LOADER_HPP

#include <vector>
#include <string>

// Structure to store image data and labels
struct DataPoint {
    int label;
    std::vector<float> pixels;
};

// Function to load data from a CSV file
std::vector<DataPoint> loadCSV(const std::string& filename);

#endif // LOADER_HPP
