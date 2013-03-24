#include "singlePerceptron.h"

SinglePerceptron::SinglePerceptron(Rand& rand, int numWeights, double learningRate)
    :learningRate(learningRate),rand(rand)
{
    if(CONIC_MODE)
    {
        if(numWeights != 2)
        {
            printf("In conic mode, you must have 2 parameters\n");
            exit(-1);
        }
        numWeights = 5;
    }
    //initialize weights
    for(int i=0;i<numWeights;i++)
    {
        //generate a random number between 1 and -1
        uint64 num = rand.next();
        double weight = num/((double)(unsigned long long)0xffffffffffffffff);
        weight = weight*2 - 1;
        weights.push_back(weight);
    }
    uint64 num = rand.next();
    biasWeight = num/((double)(unsigned long long)0xffffffffffffffff);
    biasWeight = biasWeight*2 - 1;

}

SinglePerceptron::~SinglePerceptron()
{
}

void SinglePerceptron::trainWeights(Matrix& originalData, Matrix& originalLabels)
{
    
    Matrix data;
    Matrix labels;
    labels = originalLabels;
    if(CONIC_MODE)
    {
        data = originalData;
        for(size_t r=0;r<data.rows();r++)
        {
            vector<double>& row = data.row(r);
            row.push_back(row[0]*row[0]);  //x^2
            row.push_back(row[1]*row[1]);  //y^2
            row.push_back(row[0]*row[1]);  //x*y
        }
        
    }
    else
        data = originalData;
    int epoch = 0;
    vector<int> numErrors;  //stores the errors from the last 10 epochs
    while(true)
    {
        int numberOfErrors = 0;
        epoch++;
        data.shuffleRows(rand,&labels);
        for(size_t r=0;r<data.rows();r++)
        {
            int perceptronOutput = (evaluatePerceptron(data[r])>0);
            if(perceptronOutput != labels[r][0])
            {
                numberOfErrors++;
                adjustWeights(labels[r][0],perceptronOutput,data[r]);
                //printf("output %i, target %i\n",perceptronOutput,(int)labels[r][0]);
                //for(size_t i=0;i<weights.size();i++)
                //    cout<<weights[i]<<" ";
                //cout<<endl;
            }
        }
        /*for(size_t i=0;i<weights.size();i++)
            cout<<weights[i]<<" ";
        cout<<endl;*/
        if(numErrors.size() >= 10)
        {
            double avgError = 0;
            for(size_t i=0;i<numErrors.size();i++)
                avgError += numErrors[i];
            avgError /=numErrors.size();
            if(numberOfErrors>=avgError)
                break;
            else
                numErrors.erase(numErrors.begin());
        }

        numErrors.push_back(numberOfErrors);
        //if(epoch > 1000)
        //    break;
        double percentError = numberOfErrors/((double)data.rows());
        //printf("%d,%f\n",epoch,percentError);
    }
    printf("Trained in %d epochs\n",epoch);
    for(size_t i=0;i<weights.size();i++)
        cout<<weights[i]<<" ";
    cout<<biasWeight<<endl;
}

double SinglePerceptron::evaluatePerceptron(const vector<double>& originalInputs)
{
    vector<double> inputs;
    if(CONIC_MODE)
    {
        inputs = originalInputs;
        inputs.push_back(inputs[0]*inputs[0]);
        inputs.push_back(inputs[1]*inputs[1]);
        inputs.push_back(inputs[0]*inputs[1]);
    }
    else
        inputs = originalInputs;
    double score = 0;
    for(size_t i=0;i<inputs.size();i++)
    {
        score += inputs[i]*weights[i];
    }
    score += biasWeight;

    return score;
}

void SinglePerceptron::adjustWeights(int target, int actual,vector<double>& inputs)
{
    for(size_t i=0;i<inputs.size()+1;i++)
    {
        if(i<inputs.size())
            weights[i] += learningRate*(target-actual)*inputs[i];
        else //its the bais weight
            biasWeight += learningRate*(target-actual);
    }
}



