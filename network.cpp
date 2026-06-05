#include <vector>
#include <random>
#include <iostream>
#include "network.hpp"
#include "training.hpp"

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
        this->weights = std::vector<std::vector<std::vector<double>>>(
    numberOfHiddenLayers + 1,
    std::vector<std::vector<double>>(
        hiddenSize,
        std::vector<double>(inputSize)
    )
    );
    this->biases = std::vector<std::vector<double>>(
        numberOfHiddenLayers + 1,
        std::vector<double>(hiddenSize)
    );
        for(int i = 0; i < numberOfHiddenLayers+1; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                for (int k = 0; k < weights[i][j].size(); k++) {
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

