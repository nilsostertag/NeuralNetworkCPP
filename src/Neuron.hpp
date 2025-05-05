#pragma once

#include <vector>

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned index);
    void setOutputValue(double value);
    double getOutputValue(void) const;
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetValue);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;
    static double alpha;
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double randomWeight(void);
    double sumDOW(const Layer &nextLayer) const;

    double m_outputValue;
    std::vector<Connection> m_outputWeights;
    unsigned index;
    double m_gradient;
};
