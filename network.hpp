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
    //ehh might use that line when i add functionality to save and load the model, but for now i will just use the constructor that takes the parameters

private:
    // Add private members for the neural network parameters
};

#endif // NETWORK_HPP