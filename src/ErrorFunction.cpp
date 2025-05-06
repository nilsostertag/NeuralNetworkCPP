// ErrorFunction.cpp
#include "ErrorFunction.hpp"
#include <cmath>
#include <stdexcept>

double ErrorFunction::computeError(const std::vector<double> &output, const std::vector<double> &target, ErrorType type) {
    double error = 0.0;
    switch (type) {
        case ErrorType::MeanSquaredError:
            for (size_t i = 0; i < output.size(); ++i) {
                double diff = target[i] - output[i];
                error += diff * diff;
            }
            error /= output.size();
            return error;
        case ErrorType::MeanAbsoluteError:
            for (size_t i = 0; i < output.size(); ++i) {
                error += std::abs(target[i] - output[i]);
            }
            error /= output.size();
            return error;
        default:
            throw std::invalid_argument("Unknown loss function");
    }
}

std::vector<double> ErrorFunction::computeDerivatives(const std::vector<double> &output, const std::vector<double> &target, ErrorType type) {
    std::vector<double> derivatives(output.size());

    switch (type) {
        case ErrorType::MeanSquaredError:
            for (size_t i = 0; i < output.size(); ++i) {
                derivatives[i] = output[i] - target[i];
            }
            break;
        case ErrorType::MeanAbsoluteError:
            for (size_t i = 0; i < output.size(); ++i) {
                derivatives[i] = (output[i] > target[i]) ? 1.0 : (output[i] < target[i]) ? -1.0 : 0.0;
            }
            break;
        default:
            throw std::invalid_argument("Unknown loss function");
    }

    return derivatives;
}
