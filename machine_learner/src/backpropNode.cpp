//See header

#include "backpropNode.h"

BackpropNode::BackpropNode(Rand& rand) :rand(rand)
{
    output = 0;
    error = 0;
}

BackpropNode::~BackpropNode()
{}


void BackpropNode::addInput(BackpropNode* input)
{
    this->inputs.push_back(input);
    uint64 num = rand.next();
    double weight = num/((double)(unsigned long long)0xffffffffffffffff);
    weight = weight*2 - 1;
    weights.push_back(weight);
    lastWeightChanges.push_back(0);
}

void BackpropNode::addOutput(BackpropNode* out)
{
    this->outputs.push_back(out);
}

double BackpropNode::getOutput()
{
    return output;
}

double BackpropNode::getNetValue()
{
    double sum =  0;
    for(size_t i=0;i<inputs.size();i++)
        sum += inputs[i]->getOutput()*weights[i];
    return sum;
}

double BackpropNode::getError()
{
    return error;
}

double BackpropNode::whatsMyWeight(BackpropNode* input)
{
    for(size_t i=0;i<inputs.size();i++)
    {
        if(((long)(inputs[i])) == ((long)input))
            return weights[i];
    }
    printf("Warning no weight found for node at 0x%x\n",(unsigned int)(long)input);
    return 0;
}

void BackpropNode::updateWeights()
{
    for(size_t i=0;i<weights.size();i++)
    {
        double dw = LEARNING_RATE*error*inputs[i]->getOutput();
        //add momentum
        dw += lastWeightChanges[i]*MOMENTUM;
        weights[i] += dw;
        lastWeightChanges[i] = dw;
        //printf("Node at 0x%x: dWeight from %i is %f\n",this,i,dw);
    }
}




