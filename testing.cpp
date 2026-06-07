#include <cmath>
#include "testing.hpp"

std::vector<double> predict(NeuralNetwork& network, const std::vector<double>& inputs) {
    std::vector<double> current = inputs;

    for (int i = 0; i <= network.numberOfHiddenLayers; i++) {
        int layerSize = (i == network.numberOfHiddenLayers)
            ? network.outputSize
            : network.hiddenSize;

        std::vector<double> next(layerSize);

        for (int j = 0; j < layerSize; j++) {
            double sum = 0.0;
            for (int k = 0; k < (int)current.size(); k++) {
                sum += current[k] * network.weights[i][j][k];
            }
            sum += network.biases[i][j];

            if (i == network.numberOfHiddenLayers) {
                next[j] = 1.0 / (1.0 + std::exp(-sum)); // sigmoid
            } else {
                next[j] = std::max(0.0, sum); // relu
            }
        }

        current = next;
    }

    return current;
}

std::vector<double> makeInput(double homeTeamIndex, double awayTeamIndex) {
    return { homeTeamIndex, awayTeamIndex };
}