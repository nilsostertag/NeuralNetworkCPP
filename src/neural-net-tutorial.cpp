#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {
    public:
        Neuron(unsigned numOutputs, unsigned index);
        void setOutputValue(double value) { m_outputValue = value; }
        double getOutputValue(void) const { return m_outputValue; }
        void feedForward(const Layer &prevLayer);
        void calcOutputGradients(double targetValue);
        void calcHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);

    private:
        static double eta;
        static double alpha;
        static double activationFunction(double x);
        static double activationFunctionDerivative(double x);
        static double randomWeight(void) { return rand() / double(RAND_MAX); };
        double sumDOW(const Layer &nextLayer) const;
        double m_outputValue;
        std::vector<Connection> m_outputWeights;
        unsigned index;
        double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer &prevLayer) {
    for(unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[index].deltaWeight;
        
        // IMPORTANT
        double newDeltaWeight = 
            eta
            * neuron.getOutputValue()
            * m_gradient
            * alpha
            * oldDeltaWeight;
         
        neuron.m_outputWeights[index].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[index].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;
    for(unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::activationFunctionDerivative(m_outputValue);
}

void Neuron::calcOutputGradients(double targetValue) {
    double delta = targetValue - m_outputValue;
    m_gradient = delta * Neuron::activationFunctionDerivative(m_outputValue);
}

// TODO: add multiple activation functions
double Neuron::activationFunction(double x) {
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x) {
    return 1 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;

    for(unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].m_outputValue * prevLayer[n].m_outputWeights[n].weight;
    }

    m_outputValue = activationFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned index) {
    for(unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    index = index;
}

class Net {
    public:
        Net(const std::vector<unsigned> &topology);
        void feedForward(const std::vector<double> &inputValues);
        void backPropagate(const std::vector<double> &targetValues);
        void getResults(std::vector<double> &resultValues) const;

    private:
        std::vector<Layer> m_layers; //m_layers[layerNum][neronNum]
        double m_error;
        double m_recentAverageError;
        double m_recentAverageSmoothingFactor;
};

void Net::getResults(std::vector<double> &resultValues) const {
    resultValues.clear();
    for(unsigned n = 0; n < m_layers.back().size(); ++n) {
        resultValues.push_back(m_layers.back()[n].getOutputValue());
    }
}

void Net::backPropagate(const std::vector<double> &targetValues) {
    // Calculate overall net error
    // TODO: multiple error functions (MSE, etc)
    //RMS (Root Mean Squared Error)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    for(unsigned n = 0; n < outputLayer.size(); ++n) {
        double delta = targetValues[n] - outputLayer[n].getOutputValue();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    // Implement recent average measurement
    m_recentAverageError = (m_recentAverageError* m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for(unsigned n = 0; n < outputLayer.size(); ++n) {
        outputLayer[n].calcOutputGradients(targetValues[n]);
    }

    // Calculate gradients on hidden layers 
    for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for(unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // Update connection weights for all layers from outputs to first hidden layer
    for(unsigned layerNum = m_layers.size(); layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for(unsigned n = 0; n < layer.size(); ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const std::vector<double> &inputValues) {
    assert(inputValues.size() == m_layers[0].size() - 1);
    // assign the input values to the input neurons 
    for(unsigned i = 0; i < inputValues.size(); ++i) {
        m_layers[0][i].setOutputValue(inputValues[i]);
    }

    // forward propagation
    for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const std::vector<unsigned> &topology) {
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == numLayers - 1 ? 0 : topology[layerNum + 1];

        // new layer is created, now the neurons should be initialized
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Neuron created!" << std::endl;
        }
        // force the bias value to 1
        m_layers.back().back().setOutputValue(1.0);
    }
}

int main() {
    std::vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    Net myNet(topology);

    std::vector<double> inputValues;
    myNet.feedForward(inputValues);

    std::vector<double> targetValues;
    myNet.backPropagate(targetValues);

    std::vector<double> resultValues;
    myNet.getResults(resultValues);

    // TODO: TRAINING DATA IMPORT
}