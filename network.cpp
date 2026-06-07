#include <vector>
#include <random>
#include <iostream>
#include "network.hpp"
#include "training.hpp"

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int numberOfHiddenLayers) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0); // smaller initial weights train better

    this->inputSize = inputSize;
    this->hiddenSize = hiddenSize;
    this->outputSize = outputSize;
    this->numberOfHiddenLayers = numberOfHiddenLayers;
    this->weights.resize(numberOfHiddenLayers + 1);
    this->biases.resize(numberOfHiddenLayers + 1);

    for (int i = 0; i <= numberOfHiddenLayers; i++) {
        int rows = (i == numberOfHiddenLayers) ? outputSize : hiddenSize;
        int cols = (i == 0) ? inputSize : hiddenSize;

        weights[i].assign(rows, std::vector<double>(cols));
        biases[i].assign(rows, 0.0);

        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                weights[i][j][k] = dis(gen);
            }
            biases[i][j] = dis(gen);
        }
    }

    std::cout << "Neural network initialized with input size: " << inputSize
              << ", hidden size: " << hiddenSize
              << ", output size: " << outputSize
              << ", number of hidden layers: " << numberOfHiddenLayers << std::endl;
}