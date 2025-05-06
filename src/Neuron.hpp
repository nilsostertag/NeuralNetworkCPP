//Neuron.hpp
#pragma once

#include <vector>
#include "ActivationFunction.hpp"

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
    void feedForward(const Layer &prevLayer, ActivationType activationFunction);
    void calcOutputGradients(double targetValue, ActivationType activationFunction);
    void calcHiddenGradients(const Layer &nextLayer, ActivationType activationFunction);
    void updateInputWeights(Layer &prevLayer);
    void calcOutputGradientsFromError(double errorDerivative, ActivationType activationFunction);


private:
    static double eta;
    static double alpha;
    static double execActivationFunction(double x, ActivationType activationFunction);
    static double activationFunctionDerivative(double x, ActivationType activationFunction);
    static double randomWeight(void);
    double sumDOW(const Layer &nextLayer) const;

    double m_inputSum;
    double m_outputValue;
    std::vector<Connection> m_outputWeights;
    unsigned index;
    double m_gradient;
};
