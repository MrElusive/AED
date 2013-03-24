/*
    This class manages all backprop learning.

    Author: Nathan Wonnacott
    Created: 2/11/2013
*/

#ifndef BACK_PROPAGATION_H
#define BACK_PROPAGATION_H

#include <iostream>
#include <stdio.h>
#include "learner.h"
#include "rand.h"
#include "backpropNode.h"
#include "backpropHiddenNode.h"
#include "backpropInputNode.h"
#include "backpropOutputNode.h"
#include "backpropBiasNode.h"

#define DEFAULT_NUM_HIDDEN_LAYERS   1
#define DEFAULT_NUM_HIDDEN_NODES    8

#define VALIDATION_PERCENTAGE 0.25

#define MSE_WINDOW_SIZE 25

using namespace std;

class BackPropagationLearner : public SupervisedLearner
{
private:
    Rand& rand;
    int numHiddenLayers;
    int numNodesPerHiddenLayer;
    int numOutputs;
    vector<vector<BackpropNode*> > layers;

public:
    BackPropagationLearner(Rand& r,int numLayers,int nodesPerLayer);
    virtual ~BackPropagationLearner();

    virtual void train(Matrix& features, Matrix& labels);
    virtual void predict(const vector<double>& features, vector<double>& labels);

private:
    void calculateOutputs();
    void calculateErrors();
    void updateWeights();
    void setupNet(Matrix& features, Matrix& labels);
    double findMissclass(Matrix& features,Matrix& labels);
    double getSumSquaredError(Matrix& features,Matrix& labels);
    double average(vector<double> v);

};

#endif
