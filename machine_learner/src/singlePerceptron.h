#ifndef SINGLE_PERCEPTRON_H
#define SINGLE_PERCEPTRON_H

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "rand.h"

#define CONIC_MODE true

using namespace std;

class SinglePerceptron
{
private:
    vector<double> weights;
    double biasWeight;
    double learningRate;
    Rand& rand;
public:
    /*
        Parameters:
        rand: a reference to a random number generator
        numInputs: the number of inputs (excluding the bias input)
        learningRate: the rate of how much to change weights on each iteration
    */
    SinglePerceptron(Rand& rand, int numInputs, double learningRate);
    virtual ~SinglePerceptron();

    /*
        This method runs the whole perceptron algorithm. It only needs to be 
        called once as this method handles as many epochs as needed until the 
        stopping condition is met.
    */
    void trainWeights(Matrix& data, Matrix& labels);

    /*
        This takes a set of inputs (not including the bias input) and returns
        the perceptron's output with the current weights
        Instead of returning a 0 or 1 to indicate if the number was less than
        or greater than 0, it returns the actual result before thinking about
        the 0 threshhold. This means that you can use the output as a rough 
        measure of confidence about the answer.
    */
    double evaluatePerceptron(const vector<double>& inputs);

private:
    /*
        performs the rule dwij = c(tj-zj)xi for each weight
        the inputs vector should not contain the bias input
    */
    void adjustWeights(int targetOutput, int actualOutput,vector<double>& inputs);


};

#endif


