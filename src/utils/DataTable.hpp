#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

class DataTable {
public:
    bool loadFromCSV(const std::string& filename, char delimiter = ',');
    std::vector<std::string> getColumnNames() const;
    std::vector<double> getRow(size_t index) const;
    const std::vector<double>& getColumn(const std::string& name) const;
    size_t rowCount() const;

private:
    std::vector<std::string> columnNames;
    std::unordered_map<std::string, std::vector<double>> dataSet;
    size_t numRows = 0;
};
