#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <string>

class NeuralNetwork {
public:
    int inputSize;
    int hiddenSize;
    int outputSize;
    int numberOfHiddenLayers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int numberOfHiddenLayers); 
    // NeuralNetwork(std::string modelPath);

private:
    // Add private members for the neural network parameters
};

#endif // NETWORK_HPP