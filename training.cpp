#include <numeric>
#include <cmath>
#include <iostream>
#include "network.hpp"
#include "trainingdata.hpp"

double meanSquaredError(const std::vector<double>& predicted, const std::vector<double>& target) {
    double sum = 0.0;
    size_t n = std::min(predicted.size(), target.size());
    for (size_t i = 0; i < n; i++) {
        sum += std::pow(predicted[i] - target[i], 2);
    }
    return sum / n;
}

void train(NeuralNetwork& network,
           const std::vector<TrainingSample>& data,
           double learningRate,
           int epochs) {

    std::cout << "Training neural network..." << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {

        double totalLoss = 0.0;

        for (const auto& sample : data) {

            const auto& inputs = sample.inputs;
            double target = sample.output;

            // forwared pass
            std::vector<std::vector<double>> activations;
            activations.push_back(inputs);

            std::vector<double> current = inputs;

            for (int i = 0; i <= network.numberOfHiddenLayers; i++) {

                int layerSize =
                    (i == network.numberOfHiddenLayers)
                        ? network.outputSize
                        : network.hiddenSize;

                std::vector<double> next(layerSize);

                for (int j = 0; j < layerSize; j++) {

                    double sum = 0.0;

                    for (int k = 0; k < (int)current.size(); k++) {
                        sum += current[k] * network.weights[i][j][k];
                    }

                    sum += network.biases[i][j];

                    next[j] = std::max(0.0, sum); // relu
                }

                activations.push_back(next);
                current = next;
            }

            const std::vector<double>& output = activations.back();

            double loss = std::pow(output[0] - target, 2);
            totalLoss += loss;

            // backprop
            std::vector<std::vector<double>> deltas(network.numberOfHiddenLayers + 1);

            for (int i = network.numberOfHiddenLayers; i >= 0; i--) {

                int layerSize =
                    (i == network.numberOfHiddenLayers)
                        ? network.outputSize
                        : network.hiddenSize;

                deltas[i].resize(layerSize);

                for (int j = 0; j < layerSize; j++) {

                    double activation = activations[i + 1][j];

                    double dcda;

                    if (i == network.numberOfHiddenLayers) {
                        dcda = 2.0 * (activation - target);
                    } else {
                        dcda = 0.0;

                        for (int m = 0;
                             m < (int)deltas[i + 1].size();
                             m++) {

                            dcda += deltas[i + 1][m]
                                  * network.weights[i + 1][m][j];
                        }
                    }

                    double dadz = (activation > 0.0) ? 1.0 : 0.0;
                    deltas[i][j] = dcda * dadz;
                }
            }

            // change weights and biases
            for (int i = 0; i <= network.numberOfHiddenLayers; i++) {

                int layerSize =
                    (i == network.numberOfHiddenLayers)
                        ? network.outputSize
                        : network.hiddenSize;

                for (int j = 0; j < layerSize; j++) {

                    for (int k = 0;
                         k < (int)network.weights[i][j].size();
                         k++) {

                        network.weights[i][j][k] -=
                            learningRate * deltas[i][j] * activations[i][k];
                    }

                    network.biases[i][j] -= learningRate * deltas[i][j];
                }
            }
        }

        std::cout << "Epoch " << epoch + 1
                  << "/" << epochs
                  << " Loss: " << totalLoss / data.size()
                  << std::endl;
    }
}