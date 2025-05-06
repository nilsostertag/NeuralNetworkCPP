//ActivationFunction.hpp
#pragma once

enum class ActivationType {
    Tanh,
    Sigmoid,
    ReLU
};

class ActivationFunction {
public:
    // Aktivierungsfunktion anwenden
    static double activate(double x, ActivationType type);

    // Ableitung der Aktivierungsfunktion
    static double derivative(double x, ActivationType type);
};
