#pragma once

#include <vector>
#include "network.hpp"
#include "trainingdata.hpp"

// Loss function
double meanSquaredError(const std::vector<double>& predicted,
                        const std::vector<double>& target);

// Training function (declaration only)
void train(NeuralNetwork& network,
           const std::vector<TrainingSample>& data,
           double learningRate,
           int epochs);