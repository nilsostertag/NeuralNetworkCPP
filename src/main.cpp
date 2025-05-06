//main.cpp
#include "utils/DataTable.hpp"
#include <iostream>
#include <vector>
#include "Network.hpp"

int main() {
    DataTable table;
    if (!table.loadFromCSV("/home/repos/NeuralNetworkCPP/data/trainingDataXOR.csv")) {
        std::cerr << "Konnte Datei nicht laden.\n";
        return 1;
    }
    /*
    std::cout << "Spalten:\n";
    for (const auto& name : table.getColumnNames()) {
        std::cout << " - " << name << "\n";
    }
    */
    std::vector<unsigned> topology = {2, 7, 1};
    Network myNet(topology, ActivationType::Sigmoid, ErrorType::MeanSquaredError);

    std::vector<double> inputValues, targetValues, resultValues;
    unsigned epochs = 100;
    std::vector<double> epochResults;

    for(unsigned epochCount = 0; epochCount < epochs; ++epochCount) {
        std::vector<double> tempResults;
        std::cout << "Starting training epoch " << epochCount << "/" << epochs << std::endl; 
        for(unsigned u = 0; u < table.rowCount(); ++u) {
            auto row = table.getRow(u);
            //for(auto val : row) {
            //    std::cout << val << ", ";
            //}
            //std::cout << "\n";
            inputValues = {row[0], row[1]};
            std::cout << "input: " << inputValues[0] << ".0 " << inputValues[1] << ".0" << std::endl; 
    
            myNet.feedForward(inputValues);
            myNet.getResults(resultValues);
            std::cout << "output: " << resultValues[0] << ".0 " << std::endl; 
    
            targetValues = {row[2]};
            std::cout << "target: " << targetValues[0] << ".0 " << std::endl;
            myNet.backPropagate(targetValues);
            
            double tempError = myNet.getRecentAverageError();
            std::cout << "RAE (Recent Average Error): " << tempError << std::endl;
            tempResults.push_back(tempError);
        }
        std::cout << "From: " << tempResults[0] << " to " << tempResults[tempResults.size()-1] << std::endl;
        epochResults.push_back(myNet.getRecentAverageError());
    }
    std::cout << "DONE" << std::endl;
    for(unsigned i = 0; i < epochResults.size(); ++i) {
        std::cout << "Epoch " << i + 1 << ": " << epochResults[i] << std::endl;
    }
    return 0;
}
/*
int main() {
    std::vector<unsigned> topology = {2, 4, 1};
    Network myNet(topology);

    std::vector<double> inputValues = {1.0, 0.0};
    myNet.feedForward(inputValues);

    std::vector<double> targetValues = {1.0};
    myNet.backPropagate(targetValues);

    std::vector<double> resultValues;
    myNet.getResults(resultValues);
}
*/