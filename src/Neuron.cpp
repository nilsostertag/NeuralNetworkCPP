#include "Neuron.hpp"
#include <cmath>
#include <cstdlib>

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned idx) : index(idx) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
}

void Neuron::setOutputValue(double value) {
    m_outputValue = value;
}

double Neuron::getOutputValue(void) const {
    return m_outputValue;
}

void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].m_outputValue * prevLayer[n].m_outputWeights[index].weight;
    }
    m_outputValue = activationFunction(sum);
}

void Neuron::calcOutputGradients(double targetValue) {
    double delta = targetValue - m_outputValue;
    m_gradient = delta * activationFunctionDerivative(m_outputValue);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * activationFunctionDerivative(m_outputValue);
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[index].deltaWeight;

        double newDeltaWeight =
            eta * neuron.getOutputValue() * m_gradient + alpha * oldDeltaWeight;

        neuron.m_outputWeights[index].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[index].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

double Neuron::activationFunction(double x) {
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x) {
    return 1 - x * x;
}

double Neuron::randomWeight(void) {
    return rand() / double(RAND_MAX);
}
