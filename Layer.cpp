#include <vector>

#include "Neuron.cpp";

class Layer {
    public:
    int numofNeurons;
    Perceptron neurons[numofNeurons];
    double weights[numofNeurons];
    double bias[numofNeurons];
    int layerIndex;

    public:
    Layer(int numofNeurons, std::vector<double> weights, std::vector<double> bias, int layerIndex) {
        this.numOfNeurons = numOfNeurons;
        this.layerIndex = layerIndex;
        for (size_t i = 0; i < numofNeurons; i++)
        {
            this.neurons[i] = new Percept
        }
        
    }

};

