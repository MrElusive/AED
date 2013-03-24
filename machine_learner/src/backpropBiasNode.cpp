//See header

#include "backpropBiasNode.h"

BackpropBiasNode::BackpropBiasNode(Rand& rand):BackpropNode(rand)
{
    this->output = 1;
}

BackpropBiasNode::~BackpropBiasNode()
{
}

void BackpropBiasNode::calculateOutput()
{
    //Do nothin
}

