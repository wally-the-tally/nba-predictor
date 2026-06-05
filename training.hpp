#pragma once

#include <vector>
#include "network.hpp"

// Loss function
double meanSquaredError(const std::vector<double>& predicted,
                        const std::vector<double>& target);

// Training function (declaration only)
void train(NeuralNetwork& network,
           const std::vector<std::vector<double>>& inputs,
           const std::vector<std::vector<double>>& targets,
           double learningRate,
           int epochs);