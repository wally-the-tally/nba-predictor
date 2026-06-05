#include "network.hpp"
#include "training.hpp"

int main() {
    NeuralNetwork network(3, 3, 2, 2);
    train(network, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}, {{1.0, 0.0}, {0.0, 1.0}}, 0.05, 100);
    return 0;
}