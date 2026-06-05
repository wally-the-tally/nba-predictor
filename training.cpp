#include <numeric>
#include <cmath>
#include <iostream>
#include "network.hpp"

//since im using mse as the loss function, the formula is (1/n) * sum((predicted - target)^2) where n is the number of samples
double meanSquaredError(const std::vector<double>& predicted, const std::vector<double>& target) {
    double sum = 0.0;
    size_t n = std::min(predicted.size(), target.size());
    for (size_t i = 0; i < n; i++) {
        sum += std::pow(predicted[i] - target[i], 2);
    }
    return sum / predicted.size();
}

void train(NeuralNetwork& network, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, double learningRate, int epochs) {
    // Implement the training loop here
    // For each epoch, perform forward pass, calculate loss, and update weights using backpropagation
    std::cout << "Training the neural network..." << std::endl;
    // Placeholder for training logic

    for(int inputIndex = 0; inputIndex < inputs.size(); inputIndex++) {
        //forward pass
        //go through each layer and calculate the output for every row in the input data
        std::vector<double> nextInputs = inputs[inputIndex];
        for(int i = 0; i < network.numberOfHiddenLayers + 1; i++) {
                //dot product of the inputs and the weights for the current layer and add the biases
                int layerSize = (i == network.numberOfHiddenLayers) ? network.outputSize : network.hiddenSize;
                std::vector<double> nextLayer(layerSize);
                for (int j = 0; j < layerSize; j++) {
                    double sum = 0.0;
                    for(int k = 0; k < nextInputs.size(); k++) {
                        sum += nextInputs[k] * network.weights[i][j][k];
                    }
                    sum += network.biases[i][j];
                    //apply activation function (e.g., ReLU, sigmoid, etc.)
                    // std::vector<double> nextLayer(nextInputs.size());
                    nextLayer[j] = std::max(0.0, sum); // ReLU activation
            }
            nextInputs = nextLayer; // the output of the current layer becomes the input for the next layer
        }
        // Here you would calculate the loss and perform backpropagation to update the weights and biases
        int targetIndex = inputIndex;
        double loss = meanSquaredError(nextInputs, targets[targetIndex]);
        std::cout << "Epoch " << 0 << ", Loss: " << loss << std::endl;
        //go back each layer and update the weights and biases using the loss and the learning rate
        for(int i = network.numberOfHiddenLayers; i >= 0; i--) {
            //goes through each neuron in the current layer and calculates the gradient of the loss with respect to the weights and biases
            int layerSize =(i == network.numberOfHiddenLayers)? network.outputSize: network.hiddenSize;
                for(int j = 0; j < layerSize; j++) {
                double dcda = 0.0; // derivative of the loss with respect to the output
                if (i == network.numberOfHiddenLayers) {
                    dcda = 2 * (nextInputs[j] - targets[targetIndex][j]);
                }
                double dadz = (nextInputs[j] > 0) ? 1 : 0; // derivative of the activation function (ReLU)
                double dzdw = nextInputs[j]; // derivative of the output with respect to the weights
                double gradient = dcda * dadz * dzdw; // chain rule to calculate the gradient
                //update weights and biases using the loss and the learning rate
                for(int k = 0; k < network.weights[i][j].size(); k++) {
                network.weights[i][j][k] -= learningRate * gradient * nextInputs[k];
                std::cout << "Updated weight for layer " << i << ", neuron " << j << ": " << network.weights[i][j][k] << std::endl;
                }
                network.biases[i][j] -= learningRate * gradient;
            }
        }
    }
}