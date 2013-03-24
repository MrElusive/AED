#include "decisionTreeLearner.h"


DecisionTreeLearner::DecisionTreeLearner(Rand& r):rand(r)
{
    this->head = NULL;
}

DecisionTreeLearner::~DecisionTreeLearner()
{
}

void DecisionTreeLearner::train(Matrix& features, Matrix& labels)
{
    features.shuffleRows(rand,&labels);

    //Divide into train and validate sets
    Matrix validationFeatures = features;
    Matrix validationLabels = labels;
    Matrix trainFeatures = features;
    Matrix trainLabels = labels;
    int validationSize = (int)(features.rows()*DT_VALIDATION_PERCENTAGE);    
    validationFeatures.copyPart(features,0,0,validationSize,features.cols());
    validationLabels.copyPart(labels,0,0,validationSize,1);
    trainFeatures.copyPart(features,validationSize,0,
                                    features.rows()-validationSize,
                                    features.cols());
    trainLabels.copyPart(labels,validationSize,0,
                                    labels.rows()-validationSize,
                                    1);

    //This creates the entire tree
    this->head = new DecisionTreeNode(trainFeatures,trainLabels);

    vector<DecisionTreeNode*>* allNodes = head->getAllNodes();
    //cout<<"Number of tree nodes before pruning= "<<allNodes->size()<<endl;
    /*cout<<"Accuracy on validation set = "
        <<getAccuracy(validationFeatures,validationLabels)<<endl;
    cout<<"on training set = "<<getAccuracy(trainFeatures,trainLabels)<<endl;
    cout<<endl;*/

    //prune it
    bool done = false;
    while(!done)
    {
        double bestAccuracy = getAccuracy(validationFeatures,validationLabels);
        DecisionTreeNode* nodeToBePruned = NULL;
        for(size_t i=0;i<allNodes->size();i++)
        {
            DecisionTreeNode* node = allNodes->at(i);
            if(!node->isLeaf())
            {
                node->setPruned(true);
                double accuracy = getAccuracy(validationFeatures,validationLabels);
                if(accuracy > bestAccuracy)
                {
                    nodeToBePruned = node;
                    bestAccuracy = accuracy;
                }
                node->setPruned(false);
            }
        }
        if(nodeToBePruned != NULL)
        {
            //cout<<"pruning a node"<<endl;
            nodeToBePruned->reallyPrune();
        }
        else
        {
            done = true;       
        }
    }

    //cout<<"Number of tree nodes after pruning = "<<allNodes->size()<<endl;
    /*
    cout<<"Accuracy on validation set = "
        <<getAccuracy(validationFeatures,validationLabels)<<endl;
    cout<<"on training set = "<<getAccuracy(trainFeatures,trainLabels)<<endl;
    */
    ofstream out;
    out.open("tree.dot");
    this->head->writeTree(out,3);
    out.close();

}

void DecisionTreeLearner::predict(const vector<double>& features,
                                    vector<double>& labels)
{
    //cout<<"predicting"<<endl;
    labels[0] = this->head->predict(features);
    //cout<<"predicted "<<labels[0]<<endl;
}

double DecisionTreeLearner::getAccuracy(Matrix& features, Matrix& labels)
{
    int correctPredictions = 0;
    for(size_t i=0;i<features.rows();i++)
    {
        vector<double> ans;
        ans.push_back(0);
        predict(features[i],ans);
        if(labels[i][0] == ans[0])
            correctPredictions++;
    }
    if(features.rows()==0)
        return 1;
    else
        return correctPredictions/((double)features.rows());
}




