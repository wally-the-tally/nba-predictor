#include "network.hpp"
#include "training.hpp"

int main() {
    // this is where networks are instantiated and trained and tested
    NeuralNetwork network(3,3,2,2);
        train(network, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}, {{1.0, 0.0}, {0.0, 1.0}}, 0.01, 1); // Example training call with dummy data

    return 0;
}