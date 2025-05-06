// LossFunction.hpp
#pragma once
#include <vector>

enum class ErrorType {
    MeanSquaredError,
    MeanAbsoluteError
};

class ErrorFunction {
    public:
        static double computeError(const std::vector<double> &output, const std::vector<double> &target, ErrorType type);
        static std::vector<double> computeDerivatives(const std::vector<double> &output, const std::vector<double> &target, ErrorType type);
};
