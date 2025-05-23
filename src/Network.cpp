// Network.cpp
#include "Network.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

Network::Network(const std::vector<unsigned> &topology, const ActivationType &activationFunction, const ErrorType &errorFunction) : m_error(0.0), m_recentAverageError(0.0), m_recentAverageSmoothingFactor(100.0), m_activationFunction(activationFunction), m_errorFunction(errorFunction) {
    for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1];

        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Neuron created!" << std::endl;
        }

        m_layers.back().back().setOutputValue(1.0); // bias neuron
    }
}

void Network::feedForward(const std::vector<double> &inputValues) {
    assert(inputValues.size() == m_layers[0].size() - 1);

    for (unsigned i = 0; i < inputValues.size(); ++i) {
        m_layers[0][i].setOutputValue(inputValues[i]);
    }

    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer, m_activationFunction);
        }
    }
}

void Network::backPropagate(const std::vector<double> &targetValues) {
    Layer &outputLayer = m_layers.back();

    // Ausgabevektor holen
    std::vector<double> resultValues;
    getResults(resultValues);

    // Fehlerwert berechnen über LossFunction
    m_error = sqrt(ErrorFunction::computeError(resultValues, targetValues, m_errorFunction));

    // Gleitender Durchschnitt des Fehlers
    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) /
        (m_recentAverageSmoothingFactor + 1.0);

    // Loss-Ableitungen berechnen
    std::vector<double> errorDerivatives = ErrorFunction::computeDerivatives(resultValues, targetValues, m_errorFunction);

    // Output-Gradienten mit Loss-Ableitung + Aktivierungsfunktion
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetValues[n], m_activationFunction);
    }

    // Hidden-Layer-Gradienten rückwärts berechnen
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer, m_activationFunction);
        }
    }

    // Gewichtsanpassung
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}


void Network::getResults(std::vector<double> &resultValues) const {
    resultValues.clear();
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultValues.push_back(m_layers.back()[n].getOutputValue());
    }
}
