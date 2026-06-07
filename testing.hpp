#pragma once
#include <vector>
#include "network.hpp"

std::vector<double> predict(NeuralNetwork& network, const std::vector<double>& inputs);
std::vector<double> makeInput(double homeTeamIndex, double awayTeamIndex);