#include <numeric>
#include <cmath>
#include <iostream>
#include "network.hpp"

double meanSquaredError(const std::vector<double>& predicted, const std::vector<double>& target) {
    double sum = 0.0;
    size_t n = std::min(predicted.size(), target.size());
    for (size_t i = 0; i < n; i++) {
        sum += std::pow(predicted[i] - target[i], 2);
    }
    return sum / n;
}

void train(NeuralNetwork& network, const std::vector<std::vector<double>>& inputs,
           const std::vector<std::vector<double>>& targets, double learningRate, int epochs) {

    std::cout << "Training the neural network..." << std::endl;

    // FIX: outer loop over epochs so the parameter is actually used
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;

        for (int inputIndex = 0; inputIndex < (int)inputs.size(); inputIndex++) {

            // --- FORWARD PASS ---
            // FIX: store every layer's activations — backprop needs them all,
            //      not just the final output
            std::vector<std::vector<double>> activations;
            activations.push_back(inputs[inputIndex]); // activations[0] = network input

            std::vector<double> current = inputs[inputIndex];

            for (int i = 0; i <= network.numberOfHiddenLayers; i++) {
                int layerSize = (i == network.numberOfHiddenLayers) ? network.outputSize : network.hiddenSize;
                std::vector<double> nextLayer(layerSize);

                for (int j = 0; j < layerSize; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < (int)current.size(); k++) {
                        sum += current[k] * network.weights[i][j][k];
                    }
                    sum += network.biases[i][j];
                    nextLayer[j] = std::max(0.0, sum); // ReLU
                }

                activations.push_back(nextLayer); // activations[i+1] = output of layer i
                current = nextLayer;
            }

            // activations.back() is the final network output
            const std::vector<double>& output = activations.back();
            double loss = meanSquaredError(output, targets[inputIndex]);
            totalLoss += loss;

            // --- BACKWARD PASS ---
            // Compute deltas (one per layer, from output back to first hidden)
            // delta[i] = dLoss/dz for each neuron in layer i
            std::vector<std::vector<double>> deltas(network.numberOfHiddenLayers + 1);

            for (int i = network.numberOfHiddenLayers; i >= 0; i--) {
                int layerSize = (i == network.numberOfHiddenLayers) ? network.outputSize : network.hiddenSize;
                deltas[i].resize(layerSize);

                for (int j = 0; j < layerSize; j++) {
                    double activation = activations[i + 1][j]; // output of this neuron

                    double dcda = 0.0;
                    if (i == network.numberOfHiddenLayers) {
                        // Output layer: dLoss/da = 2*(output - target) / n
                        dcda = 2.0 * (activation - targets[inputIndex][j]) / output.size();
                    } else {
                        // Hidden layer: propagate delta from the layer above
                        int nextLayerSize = (i + 1 == network.numberOfHiddenLayers) ? network.outputSize : network.hiddenSize;
                        for (int m = 0; m < nextLayerSize; m++) {
                            dcda += deltas[i + 1][m] * network.weights[i + 1][m][j];
                        }
                    }

                    // FIX: ReLU derivative uses pre-activation value; since we stored
                    //      post-activation we approximate: active iff activation > 0
                    double dadz = (activation > 0.0) ? 1.0 : 0.0;
                    deltas[i][j] = dcda * dadz;
                }
            }

            // Apply weight and bias updates
            for (int i = 0; i <= network.numberOfHiddenLayers; i++) {
                int layerSize = (i == network.numberOfHiddenLayers) ? network.outputSize : network.hiddenSize;

                for (int j = 0; j < layerSize; j++) {
                    // FIX: dzdw is the input to this neuron — activations[i], not activations[i+1]
                    for (int k = 0; k < (int)network.weights[i][j].size(); k++) {
                        network.weights[i][j][k] -= learningRate * deltas[i][j] * activations[i][k];
                    }
                    network.biases[i][j] -= learningRate * deltas[i][j];
                }
            }
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Loss: " << totalLoss / inputs.size() << std::endl;
    }
}