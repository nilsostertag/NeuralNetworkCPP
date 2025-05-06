//Neuron.cpp
#include "Neuron.hpp"
#include <cmath>
#include <cstdlib>
#include <string>

double Neuron::eta = 0.05;
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

void Neuron::feedForward(const Layer &prevLayer, ActivationType activationFunction) {
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].m_outputValue * prevLayer[n].m_outputWeights[index].weight;
    }
    m_inputSum = sum;
    m_outputValue = execActivationFunction(sum, activationFunction);
}

void Neuron::calcOutputGradients(double targetValue, ActivationType activationFunction) {
    double delta = targetValue - m_outputValue;
    m_gradient = delta * activationFunctionDerivative(m_inputSum, activationFunction);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer, ActivationType activationFunction) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * activationFunctionDerivative(m_inputSum, activationFunction);
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

double Neuron::execActivationFunction(double x, ActivationType activationFunction) {
    double result = ActivationFunction::activate(x, activationFunction);
    return result;
}

double Neuron::activationFunctionDerivative(double x, ActivationType activationFunction) {
    double result = ActivationFunction::derivative(x, activationFunction);
    return result;
}

double Neuron::randomWeight(void) {
    return rand() / double(RAND_MAX);
    //return (rand() / double(RAND_MAX)) * 2.0 - 1.0; // Bereich: [-1.0, +1.0]
}

void Neuron::calcOutputGradientsFromError(double errorDerivative, ActivationType activationFunction) {
    m_gradient = errorDerivative * activationFunctionDerivative(m_inputSum, activationFunction);
}