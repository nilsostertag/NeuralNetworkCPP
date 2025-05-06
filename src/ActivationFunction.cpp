//ActivationFunction.cpp
#include "ActivationFunction.hpp"
#include <cmath>
#include <stdexcept>

double ActivationFunction::activate(double x, ActivationType type) {
    switch (type) {
        case ActivationType::Tanh:
            return std::tanh(x);
        case ActivationType::Sigmoid:
            return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::ReLU:
            return x > 0 ? x : 0;
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

double ActivationFunction::derivative(double x, ActivationType type) {
    switch (type) {
        case ActivationType::Tanh: {
            double th = std::tanh(x);
            return 1.0 - th * th;
        }
        case ActivationType::Sigmoid: {
            double sig = 1.0 / (1.0 + std::exp(-x));
            return sig * (1.0 - sig);
        }
        case ActivationType::ReLU:
            return x > 0 ? 1.0 : 0.0;
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}
