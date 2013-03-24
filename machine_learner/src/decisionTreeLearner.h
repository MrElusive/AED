/*
    Implements the decision tree learning algorithm

    Author: Nathan Wonnacott
    Created: 1/25/2013
*/

#ifndef DECISION_TREE_LEARNER
#define DECISION_TREE_LEARNER

#include <iostream>
#include <stdio.h>
#include "learner.h"
#include "rand.h"
#include "decisionTreeNode.h"


#define DT_VALIDATION_PERCENTAGE 0.3

using namespace std;

class DecisionTreeLearner : public SupervisedLearner
{
private:
    Rand& rand;
    DecisionTreeNode* head;

public:
    DecisionTreeLearner(Rand& r);
    virtual ~DecisionTreeLearner();

    virtual void train(Matrix& features, Matrix& labels);
    virtual void predict(const vector<double>& features, vector<double>& labels);

private:
    double getAccuracy(Matrix& features, Matrix& labels);

};

#endif


