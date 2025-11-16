#pragma once
#include <vector>
#include "CTRNN.h"

class RobotController {
public:
    CTRNN net;
    int input_dim;
    int output_dim;
    double dt;

    RobotController(int n, int out, double step)
        : net(n), input_dim(n), output_dim(out), dt(step)
    {
        net.SetCircuitSize(n);
        net.RandomizeCircuitState(-0.5, 0.5);
    }

    void reset() {
        net.RandomizeCircuitState(-0.5, 0.5);
    }

    void loadParameters(const std::vector<double>& p) {
        int N = net.size;
        int idx = 0;

        // weights
        for(int i=1;i<=N;i++)
            for(int j=1;j<=N;j++)
                net.SetConnectionWeight(i, j, p[idx++]);

        // biases
        for(int i=1;i<=N;i++)
            net.SetNeuronBias(i, p[idx++]);

        // gains
        for(int i=1;i<=N;i++)
            net.SetNeuronGain(i, p[idx++]);
    }

    std::vector<double> step(const std::vector<double>& sensors)
    {
        int N = net.size;

        for(int i=1;i<=input_dim;i++){
            double s = sensors[i % sensors.size()];
            net.SetNeuronExternalInput(i, s);
        }

        net.EulerStep(dt);

        std::vector<double> out(output_dim);
        for(int i=0;i<output_dim;i++){
            out[i] = net.NeuronOutput(N - output_dim + i);
        }
        return out;
    }
};