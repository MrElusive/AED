/*
    This class manages all backprop learning.

    Author: Nathan Wonnacott
    Created: 2/11/2013
*/

#include "backpropagation.h"
#include <typeinfo>


BackPropagationLearner::BackPropagationLearner(Rand& r,int layers, int nodesPerLayer)
    :SupervisedLearner(), rand(r)
{
    numHiddenLayers = layers;
    numNodesPerHiddenLayer = nodesPerLayer;
    cout<<"Creating Backrop with "<<layers<<" hidden layers and "
        <<nodesPerLayer<<" hidden nodes per layer"<<endl;
    cout<<"Learning rate:"<<LEARNING_RATE<<endl;
    cout<<"Momentum:"<<MOMENTUM<<endl;

}
BackPropagationLearner::~BackPropagationLearner()
{
    //clean up
    for(size_t i=0;i<layers.size();i++)
    {
        for(size_t j=0;j<layers[i].size();j++)
            delete layers[i][j];
    }
}

void BackPropagationLearner::train(Matrix& features, Matrix& labels)
{
    setupNet(features,labels);
    features.shuffleRows(rand,&labels);

    //Divide into train and validate sets
    Matrix validationFeatures = features;
    Matrix validationLabels = labels;
    Matrix trainFeatures = features;
    Matrix trainLabels = labels;
    int validationSize = (int)(features.rows()*VALIDATION_PERCENTAGE);    
    validationFeatures.copyPart(features,0,0,validationSize,features.cols());
    validationLabels.copyPart(labels,0,0,validationSize,1);
    trainFeatures.copyPart(features,validationSize+1,0,
                                    features.rows()-validationSize - 1,
                                    features.cols());
    trainLabels.copyPart(labels,validationSize+1,0,
                                    labels.rows()-validationSize -1,
                                    1);
    //Do the actual training
    int epoch = 1;
    vector<double> currentErrorWindow;
    vector<double> previousErrorWindow;
    for(int i=0;i<MSE_WINDOW_SIZE;i++)
    {
        currentErrorWindow.push_back(100000000);
        previousErrorWindow.push_back(1000000000);
    }

    for(;;)
    {
        trainFeatures.shuffleRows(rand,&trainLabels);
        //for each entry
        for(size_t r=0;r<trainFeatures.rows();r++)
        {
            //printf("\ntraining element #%d\n",r);
            //set the inputs to the inputs in the features
            for(size_t i=0;i<trainFeatures.cols();i++)
                ((BackpropInputNode*)(layers[0][i]))->
                            setInputValue(trainFeatures[r][i]);
            
            //set the expected targets            
            int target = (int)trainLabels[r][0];
            vector<BackpropNode*> outputs = layers[layers.size()-1];
            if(numOutputs==1)
            {
                BackpropOutputNode* n = (BackpropOutputNode*)outputs[0];
                n->setExpectedOutput(target);
            }
            else
            {
                for(int o=0;o<numOutputs;o++)
                {
                    BackpropOutputNode* n = (BackpropOutputNode*)outputs[o];
                    n->setExpectedOutput(o==target);
                }
            }
            //Now actually do the training
            calculateOutputs();
            calculateErrors();
            updateWeights();

        }
        double missclassification = findMissclass(validationFeatures,
                                                        validationLabels);
        double sumSquaredError = getSumSquaredError(validationFeatures,
                                                        validationLabels);
       
        //printf("epoch,%i,missClass,%f,SSE,%f",
        //        epoch, missclassification,sumSquaredError);
        double trainMisClas = findMissclass(trainFeatures,trainLabels);
        double trainSSE = getSumSquaredError(trainFeatures,trainLabels);
        /*printf("%i,Train MSE,%f,validation MSE,%f\n",
                epoch,
                trainSSE/trainFeatures.rows(),
                sumSquaredError/validationFeatures.rows());
        */
        

        //Adjust the error windows
        previousErrorWindow.erase(previousErrorWindow.begin());
        previousErrorWindow.push_back(currentErrorWindow[0]);
        currentErrorWindow.erase(currentErrorWindow.begin());
        currentErrorWindow.push_back(missclassification);

        if((average(previousErrorWindow)-average(currentErrorWindow))<0.0001)
            break;
        //if(epoch>100)
        //    break;
        epoch++;
    }
}
void BackPropagationLearner::predict(const vector<double>& features, 
                                    vector<double>& labels)
{
    //set the inputs to the inputs in the features
    for(size_t i=0;i<features.size();i++)
        ((BackpropInputNode*)(layers[0][i]))->setInputValue(features[i]);

    calculateOutputs();

    if(numOutputs == 1)
    {
        BackpropOutputNode* node = (BackpropOutputNode*)layers[layers.size()-1][0];
        labels[0] = (node->getOutput()>0.5);
    }
    else
    {
        double maxVal = 0;
        for(int i=0;i<numOutputs;i++)
        {
            double val = layers[layers.size()-1][i]->getOutput();
            //printf("prediction[%i]->%f\n",i,val);
            if(val > maxVal)
            {
                labels[0] = i;
                maxVal = val;
            }
        }
    }
}


double BackPropagationLearner::findMissclass(Matrix& features,Matrix& labels)
{
    int numElements = features.rows();
    int numErrors = 0;
    for(size_t i=0;i<features.rows();i++)
    {
        vector<double> predictedLabel;
        predictedLabel.push_back(0);
        this->predict(features.row(i),predictedLabel);
        if(predictedLabel[0] != labels[i][0])
            numErrors++;
    }

    return ((double)numErrors)/numElements;
}

double BackPropagationLearner::getSumSquaredError(Matrix& features,Matrix& labels)
{
    double sumSquaredError=0;
    vector<BackpropOutputNode*> outputs;
    for(int i=0;i<numOutputs;i++)
        outputs.push_back((BackpropOutputNode*)layers[layers.size()-1][i]);

    for(size_t i=0;i<features.rows();i++)
    {
        //Set up inputs
        for(size_t a=0;a<features.cols();a++)
                ((BackpropInputNode*)(layers[0][a]))->
                            setInputValue(features[i][a]);
        //set expectedOutputs
        if(numOutputs == 1)
            outputs[0]->setExpectedOutput(labels[i][0]);
        else
            for(size_t o=0;o<outputs.size();o++)
                outputs[o]->setExpectedOutput(labels[i][0]==o);
        calculateOutputs();
        for(size_t o=0;o<outputs.size();o++)
            sumSquaredError += ((BackpropOutputNode*)outputs[o])
                                ->getLastSquaredError();
    }
    return sumSquaredError;
}

void BackPropagationLearner::setupNet(Matrix& features,Matrix& labels)
{
    //setup input layer
    vector<BackpropNode*> inputs;
    for(size_t i=0;i<features.cols();i++)
    {
        inputs.push_back(new BackpropInputNode(rand));
    }
    layers.push_back(inputs);

    //setup hidden layers
    for(int l=0;l<numHiddenLayers;l++)
    {
        vector<BackpropNode*> layer;
        for(int n=0;n<numNodesPerHiddenLayer;n++)
        {
            layer.push_back(new BackpropHiddenNode(rand));
        }
        layers.push_back(layer);
    }

    //setup output layer
    numOutputs = labels.valueCount(0);
    if(numOutputs==2)
        numOutputs = 1;
    printf("numOutputs = %i\n",numOutputs);
    vector<BackpropNode*> outputs;
    for(int i=0;i<numOutputs;i++)
    {   
        BackpropOutputNode* node = new BackpropOutputNode(rand);
        outputs.push_back(node);
    }
    layers.push_back(outputs);

    //make layers fully connected and add bias nodes
    for(size_t l=0; l< layers.size()-1;l++)
    {
        layers[l].push_back(new BackpropBiasNode(rand));
        for(size_t c=0;c<layers[l].size();c++)
        {
            BackpropNode* child = layers[l][c];
            for(size_t p=0;p<layers[l+1].size();p++)
            {
                layers[l+1][p]->addInput(child);
                child->addOutput(layers[l+1][p]);
            }
        }
    }

    /*for(size_t l=0; l< layers.size();l++)
    {
        printf("layer[%d]:\t",l);
        for(size_t n=0; n<layers[l].size();n++)
        {
            printf("%d at 0x%x,\t",n,layers[l][n]);
        }
        cout<<endl;
    }*/
}


void BackPropagationLearner::calculateOutputs()
{
    //Go from the bottom to the top calculating each output
    for(size_t l=0;l<layers.size();l++)
    {
        vector<BackpropNode*> layer = layers[l];
        for(size_t n=0;n<layer.size();n++)
            layer[n]->calculateOutput();
    }
}

void BackPropagationLearner::calculateErrors()
{
    //Go from the top to bottom calculating each error
    for(int l= (int)layers.size()-1;l>=0;l--)
    {
        vector<BackpropNode*> layer = layers[l];
        for(size_t n=0;n<layer.size();n++)
            layer[n]->calculateError();
    }
}

void BackPropagationLearner::updateWeights()
{
    //Go from the bottom to the top calculating each output
    //On this one it doesn't matter if its top to bottom or bottom to top
    //you can skip the very first layer because they don't have weights
    for(size_t l=1;l<layers.size();l++)
    {
        vector<BackpropNode*> layer = layers[l];
        for(size_t n=0;n<layer.size();n++)
            layer[n]->updateWeights();
    }
}

double BackPropagationLearner::average(vector<double> v)
{
    int size = (int)v.size();
    double total = 0;
    for(size_t i=0;i<v.size();i++)
        total += v[i];
    return total/size;
}





