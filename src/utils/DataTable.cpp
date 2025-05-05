#include "DataTable.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

bool DataTable::loadFromCSV(const std::string& filename, char delimiter) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    // First row are column names
    if (!std::getline(file, line)) return false;

    std::stringstream ss(line);
    std::string column;
    while (std::getline(ss, column, delimiter)) {
        columnNames.push_back(column);
        dataSet[column] = std::vector<double>();
    }

    // Other lines are data
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        size_t colIdx = 0;

        while (std::getline(lineStream, cell, delimiter)) {
            if (colIdx >= columnNames.size()) throw std::runtime_error("Mehr Werte als Spalten!");
            double value = std::stof(cell);
            dataSet[columnNames[colIdx]].push_back(value);
            ++colIdx;
        }
        if (colIdx != columnNames.size()) throw std::runtime_error("Zu wenige Werte in Zeile!");
        ++numRows;
    }

    return true;
}

std::vector<std::string> DataTable::getColumnNames() const {
    return columnNames;
}

std::vector<double> DataTable::getRow(size_t index) const {
    if (index >= numRows) throw std::out_of_range("Ungültiger Zeilenindex");

    std::vector<double> row;
    for (const auto& col : columnNames) {
        row.push_back(dataSet.at(col)[index]);
    }
    return row;
}

const std::vector<double>& DataTable::getColumn(const std::string& name) const {
    auto it = dataSet.find(name);
    if (it == dataSet.end()) throw std::invalid_argument("Spalte nicht gefunden: " + name);
    return it->second;
}

size_t DataTable::rowCount() const {
    return numRows;
}
