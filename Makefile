# Compilers and Flags
CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++17

# Files
SRCS = src/main.cpp src/utils/DataTable.cpp src/Network.cpp src/Neuron.cpp src/ActivationFunction.cpp src/ErrorFunction.cpp
OBJS = $(SRCS:.cpp=.o)
DEPS = src/utils/DataTable.hpp src/Network.hpp src/Neuron.hpp src/ActivationFunction.hpp src/ErrorFunction.hpp

# Target-Binary
TARGET = NeuralNetwork

# Save location
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
