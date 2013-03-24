//Handles K nearest neighbor instance based learning

#ifndef KNN_LEARNER_H
#define KNN_LEARNER_H

#define REDUCE
#define DISTANCE_WEIGHTING
//#define ALL_K

#include <set>
#include <utility>  //has pair
#include <map>
#include <iostream>     // std::cout
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include "learner.h"
#include "rand.h"
#include "knnInstance.h"

using namespace std;

//Comparer for the distances set
struct compareDistance
{
    bool operator()(const pair<int,double> p1, const pair<int, double> p2)
    {

        if(p1.second<p2.second)
            return true;
        else if(p1.second>p2.second)
            return false;
        else if(p1.first<p2.first)
            return true;
        else
            return false;
    }
};

class KNN_Learner : public SupervisedLearner
{
private:
    Rand& rand;
    int k;
    int callsToPredict; //just for debug outputs
    vector<KNN_Instance*> points;
    bool regressionMode;    //true iff we are doing regression (set in train())
public:
    KNN_Learner(Rand& r, int k);
    virtual ~KNN_Learner();

    virtual void train(Matrix& features, Matrix& labels);
    virtual void predict(const vector<double>& features, vector<double>& labels);

private:
    double voteNonRegression(set<pair<int,double>,compareDistance>& distances);
    double voteRegression(set<pair<int,double>,compareDistance>& distances);
    /*
        uses leave one out style reduction to prune points
    */
    void reduce(); 

    //returns true if |n1-n2|<=tolerance
    bool almostEqual(double n1, double n2, double tolerance);
};

#endif


