/*
    Inherits from backpropNode. Represents an output node.
*/

#ifndef BACK_PROP_OUTPUT_NODE
#define BACK_PROP_OUTPUT_NODE

#include "backpropNode.h"
#include <math.h>

class BackpropOutputNode : public BackpropNode
{
private:
    double expectedOutput;

public:
    BackpropOutputNode(Rand& rand);
    virtual ~BackpropOutputNode();
    
    virtual void calculateOutput();
    virtual void calculateError();

    void setExpectedOutput(double num);
    double getLastSquaredError();

};

#endif

