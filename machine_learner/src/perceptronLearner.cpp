#include "perceptronLearner.h"

PerceptronLearner::PerceptronLearner(Rand& r)
    :SupervisedLearner(), rand(r)
{
    perceptrons = NULL;
    numberOfPerceptrons = 0;
}
PerceptronLearner::~PerceptronLearner()
{
    for(int i=0;i<numberOfPerceptrons;i++)
        delete perceptrons[i];
    delete[] perceptrons;
}

void PerceptronLearner::train(Matrix& features, Matrix& labels)
{
    features.shuffleRows(rand,&labels);
    printf("original data set\n");
    for(size_t r=0;r<features.rows();r++)
    {
        //cout<<r<<": ";
        for(size_t c=0;c<features.cols();c++)
            cout<<features[r][c]<<", ";
        //cout<<"->";
        for(size_t c=0;c<labels.cols();c++)
            cout<<labels[r][c]<<", ";
        cout<<endl;
    }

    numberOfPerceptrons = labels.valueCount(0);
    if(numberOfPerceptrons == 2)
        numberOfPerceptrons =1;

    perceptrons = new SinglePerceptron*[numberOfPerceptrons];

    if(numberOfPerceptrons == 1)
    {
        perceptrons[0] = new SinglePerceptron(rand,features.cols(),0.1);
        perceptrons[0]->trainWeights(features,labels);
    }
    else
    {
        
        for(int i=0;i<numberOfPerceptrons;i++)
        {
            printf("perceptron %i\n",i);
            Matrix* adjustedLabels;
            adjustedLabels = labels.createCopyForBinaryPerceptron(i);
            //Use a copy of the features because the training will shuffle
            //and we don't want the features to shuffle independent of the
            //labels
            Matrix featuresCopy;
            featuresCopy = features;
            perceptrons[i] = new SinglePerceptron(rand,features.cols(),0.1);
            perceptrons[i]->trainWeights(featuresCopy,*adjustedLabels);
            delete adjustedLabels;
        }
    }
}
void PerceptronLearner::predict(const vector<double>& features, vector<double>& labels)
{
    if(numberOfPerceptrons == 1)
        labels[0] = (perceptrons[0]->evaluatePerceptron(features)>0);
    else
    {
        double maxAnswer = -1000000;
        double maxAnswerNumber = 0;
        
        for(int i=0;i<numberOfPerceptrons;i++)
        {
            double answer = perceptrons[i]->evaluatePerceptron(features);
            if(answer > maxAnswer)
            {
                maxAnswer = answer;
                maxAnswerNumber = i;
            }

        }
        labels[0] = maxAnswerNumber;

    }
}



