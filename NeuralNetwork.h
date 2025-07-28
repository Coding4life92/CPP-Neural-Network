#pragma once
#include "Matrix.h"
#include <cstdlib>

inline float Sigmoid(float x)
{
    return 1.0 / (1 + exp(-x));
}

inline float DSigmoid(float x)
{
    return x * (1 - x);
}

class SimpleNeuralNetwork
{
private:
    std::vector<uint32_t> _topology;
    std::vector<Matrix> _weightMatrices;
    std::vector<Matrix> _valueMatrices;
    std::vector<Matrix> _biasMatrices;
    float _learningRate;

public:
    SimpleNeuralNetwork(std::vector<uint32_t> topology, float learningRate = 0.1f)
        : _topology(topology),
          _weightMatrices({}),
          _valueMatrices({}),
          _biasMatrices({}),
          _learningRate(learningRate)

    {
        for (uint32_t i = 0; i < topology.size() - 1; i++)
        {
            Matrix weightMatrix(topology[i + 1], topology[i]);
            weightMatrix =
                weightMatrix.applyFunction(
                    [](const float &f)
                    {
                        return ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // [-1, 1]
                    });

            _weightMatrices.push_back(weightMatrix);
            Matrix biasMatrix(topology[i + 1], 1);
            biasMatrix = biasMatrix.applyFunction(
                [](const float &f)
                {
                    return (float)rand() / RAND_MAX;
                });
            _biasMatrices.push_back(biasMatrix);
        }
        _valueMatrices.resize(topology.size());
    }

    bool feedForward(std::vector<float> input)
    {
        if (input.size() != _topology[0])
            return false;

        Matrix values(input.size(), 1);
        for (uint32_t i = 0; i < input.size(); i++)
        {
            values._vals[i] = input[i];
        }

        // feed forward to next layer
        for (uint32_t i = 0; i < _weightMatrices.size(); i++)
        {
            _valueMatrices[i] = values;
            values = values.multiply(_weightMatrices[i]);
            values = values.add(_biasMatrices[i]);
            values = values.applyFunction(Sigmoid);
        }

        _valueMatrices[_weightMatrices.size()] = values;

        return true;
    }

    bool backPropagate(std::vector<float> targetOutput)
    {
        if (targetOutput.size() != _topology.back())
            return false;
        Matrix errors(targetOutput.size(), 1);
        errors._vals = targetOutput;
        Matrix sub = _valueMatrices.back().negative();
        errors = errors.add(sub);

        for (int i = _weightMatrices.size() - 1; i >= 0; i--)
        {
            Matrix trans = _weightMatrices[i].transpose();

            Matrix prevErros = errors.multiply(trans);
            Matrix dOutputs = _valueMatrices[i + 1].applyFunction(
                DSigmoid);

            Matrix gradients = errors.multiplyElement(dOutputs);
            gradients = gradients.multiplyScalar(_learningRate);
            Matrix weightGradients = _valueMatrices[i].transpose().multiply(gradients);

            _weightMatrices[i] = _weightMatrices[i].add(weightGradients);
            _biasMatrices[i] = _biasMatrices[i].add(gradients);
            errors = prevErros;
        }

        return true;
    }

    std::vector<float> getPrediction()
    {
        return _valueMatrices.back()._vals;
    }
};