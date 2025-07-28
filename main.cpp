#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include "NeuralNetwork.h"

std::vector<float> draw_0()
{
    return {
        0, 1, 1, 1, 1, 1, 0,
        1, 1, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 1,
        1, 1, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 0};
}

std::vector<float> draw_1()
{
    return {
        0, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 0, 0,
        0, 1, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 1, 1};
}

std::vector<float> draw_2()
{
    return {
        0, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1};
}

std::vector<float> draw_3()
{
    return {
        1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 0,
        1, 0, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 0};
}

std::vector<float> draw_4()
{
    return {
        0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 0, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 0};
}

std::vector<float> draw_5()
{
    return {
        1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 0};
}

std::vector<float> draw_6()
{
    return {
        0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 0,
        1, 1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 0};
}

std::vector<float> draw_7()
{
    return {
        1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0};
}

std::vector<float> draw_8()
{
    return {
        0, 1, 1, 1, 1, 1, 0,
        1, 1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 0,
        1, 1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 0};
}

std::vector<float> draw_9()
{
    return {
        0, 1, 1, 1, 1, 1, 0,
        1, 1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0};
}

int main()
{
    // Network topology: input = 1, output = 49 (7x7)
    std::vector<uint32_t> topology = {1, 64, 64, 49};
    SimpleNeuralNetwork nn(topology, 0.01f);

    std::vector<std::vector<float>> inputs = {
        {0.0f / 9}, {1.0f / 9}, {2.0f / 9}, {3.0f / 9}, {4.0f / 9}, {5.0f / 9}, {6.0f / 9}, {7.0f / 9}, {8.0f / 9}, {9.0f / 9}};

    std::vector<std::vector<float>> outputs = {
        draw_0(), draw_1(), draw_2(), draw_3(), draw_4(),
        draw_5(), draw_6(), draw_7(), draw_8(), draw_9()};

    std::cout << "Training started...\n";

    for (int epoch = 0; epoch < 100000; ++epoch)
    {
        int idx = rand() % 10;
        nn.feedForward(inputs[idx]);
        nn.backPropagate(outputs[idx]);
    }

    std::cout << "Training complete!\n\n";

    float digit = 4.0f / 9;
    nn.feedForward({digit});
    auto prediction = nn.getPrediction();

    std::cout << "Predicted digit drawing for: " << digit << "\n";
    for (int y = 0; y < 7; ++y)
    {
        for (int x = 0; x < 7; ++x)
        {
            float val = prediction[y * 7 + x];
            std::cout << (val > 0.5f ? '#' : ' ');
        }
        std::cout << "\n";
    }

    return 0;
}
