/*
    Represents one node or division in the decision tree
*/

#ifndef DECISION_TREE_NODE_H
#define DECISION_TREE_NODE_H

#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "matrix.h"

using namespace std;

#define ID3
//#define LAPLACIAN
//#define RETURN_0

class DecisionTreeNode
{
private:
    /*
        This keeps track of all of the nodes in the whole tree.
        This is used so that you can iteratively try every node
    */
    vector<DecisionTreeNode*>* allNodes;
    vector<DecisionTreeNode*> children;
    //This is a parrallel array to the children array that keeps
    //track of what value each child represents
    vector<double> childrenValues;
    Matrix features;
    Matrix labels;
    //This is the attribute that your children split on
    //note that it is an index into your features matrix, not the original
    //features matrix
    size_t  split;
    bool isHead;    //The head manages the allNodes vector
    bool isPruned;
    string name;
    
public:
    //This is to construct the head node
    DecisionTreeNode(Matrix& features, Matrix& labels);
    //This is to construct all of the non-head nodes
    DecisionTreeNode(Matrix& features, Matrix& labels, vector<DecisionTreeNode*>* all);
    ~DecisionTreeNode();
    
    double predict(vector<double> features);

    vector<DecisionTreeNode*>* getAllNodes();

    bool isLeaf();

    /*
        When pruned is true, this node acts as if all of its children are
        removed and predict just reports the majority class.
    */
    void setPruned(bool pruned);

    /*
        Actually removes all children branches
    */
    void reallyPrune();

    /*
        optional parameter layers, how many layers of tree to
        print. Any negative number means all layers
    */
    void writeTree(ofstream& out, int layers =-1);

    void setName(string);
    string getName();

private:
    //Bot constructors call this. This is where the actual construction happens
    void init(Matrix& features, Matrix& labels);
    bool shouldWeContinueSplitting();
    size_t getIndexOfNextBestSplit();

    /*
        Returns the info for the given attribute
    */
    double getInfoForAttr(size_t attr, double value);
};

#endif


