#include <iostream>

#include "network.hpp"
#include "training.hpp"
#include "trainingdata.hpp"
#include "testing.hpp"

int main() {
    std::vector<TrainingSample> trainingData = generateTrainingData();
    //1610612752 knicks team 15
    //1610612759 spurs team 22
    NeuralNetwork network(2, 3, 1, 3);
    auto input = makeInput(15.0, 22.0); // Example: Knicks vs Spurs
    train(network, trainingData, 0.0001, 50);
    auto output = predict(network, input);
    std::cout << "Win probability: " << output[0] << std::endl;
    return 0;
}