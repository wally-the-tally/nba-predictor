#include <vector>
#include <random>
#include <iostream>
#include "network.hpp"
// public:
//     int inputSize;
//     int hiddenSize;
//     int outputSize;
//     int numberOfHiddenLayers;
// public:
NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int numberOfHiddenLayers) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-100.0, 100.0);
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;
        this->numberOfHiddenLayers = numberOfHiddenLayers;
        // Initialize the neural network parameters
        std::vector<std::vector<double>> weights(numberOfHiddenLayers+1, std::vector<double>(hiddenSize, 0.0));
        std::vector<std::vector<double>> biases(numberOfHiddenLayers+1, std::vector<double>(hiddenSize, 0.0));
        for(int i = 0; i < numberOfHiddenLayers+1; i++) {
            std::cout << "Layer " << i << " weights: ";
            for (double weight : weights[i]) {
                weight = dis(gen);
                std::cout << weight << " ";
            }
            std::cout << "\nLayer " << i << " biases: ";
            for (double bias : biases[i]) {
                bias = dis(gen);
                std::cout << bias << " ";
            }
            std::cout << std::endl;
        }
    }

