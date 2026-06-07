#pragma once

#include <vector>
#include <string>

struct TrainingSample {
    std::vector<double> inputs;
    double output;
};

// Loads and parses Games.csv into training samples
std::vector<TrainingSample> generateTrainingData();