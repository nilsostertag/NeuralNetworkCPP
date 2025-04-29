// Net.hpp
#pragma once

#include <vector>
#include "Neuron.hpp"

class Network {
public:
    Network(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputValues);
    void backPropagate(const std::vector<double> &targetValues);
    void getResults(std::vector<double> &resultValues) const;
    double getRecentAverageError() {return m_recentAverageError; };

private:
    std::vector<Layer> m_layers;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};
