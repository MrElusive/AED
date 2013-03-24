/*
    Inherits from backpropNode. Represents an input node.
*/

#ifndef BACK_PROP_INPUT_NODE
#define BACK_PROP_INPUT_NODE

#include "backpropNode.h"

class BackpropInputNode : public BackpropNode
{
public:
    BackpropInputNode(Rand& rand);
    virtual ~BackpropInputNode();

    void setInputValue(double val);

    virtual void calculateOutput();
};

#endif

