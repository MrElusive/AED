import os

os.system("ls outputs/*output.txt > outputs/outputList.txt")

outputs = open("outputs/outputList.txt").readlines()
outputs = map((lambda x:x[0:len(x)-1]),outputs)

#This is a dictionary of dictionaries
accuracies = {}
learningRate = 0
momentum = 0
for o in outputs:
    print "Parsing file", o
    f = open(o,'r')
    lines = f.readlines()
    lines = map((lambda x:x[0:len(x)-1]),lines)
    currentSet = "None"
    setAccuracies = {}
    for l in lines:
        #print l
        if "Accuracy on the test set:" in l:
            currentSet = "test"
        elif "Accuracy on the training set:" in l:
            currentSet = "train"
        elif "Set accuracy" in l:
            accuracy = (float)(l[l.find(':')+1:len(l)])
            setAccuracies[currentSet] = accuracy
        elif "Learning rate"  in l:
            learningRate = (float)(l[l.find(':')+1:len(l)])
            setAccuracies["learningRate"] = learningRate
        elif "Momentum" in l:
            momentum = (float)(l[l.find(':')+1:len(l)])
            setAccuracies["momentum"] = momentum
    accuracies[o] = setAccuracies
        
    f.close()
resultsFile = "outputs/results_"+str(learningRate)+"_"+str(momentum)+".csv"
results = open(resultsFile,'w')
results.write("Arff File, Training Accuracy, Test Accuracy, Learning Rate, Momentum\n")

totalTestAccuracy = 0
numTestAccuracies = 0

for arff in accuracies.keys():
    print "Writing data for",arff
    print accuracies[arff]
    results.write(arff+",")
    if "train" in accuracies[arff].keys():
        results.write(str(accuracies[arff]["train"]))
    results.write(",")
    if "test" in accuracies[arff].keys():
        results.write(str(accuracies[arff]["test"]))
        totalTestAccuracy += accuracies[arff]["test"]
        numTestAccuracies += 1
    results.write(",")
    if "learningRate" in accuracies[arff].keys():
        results.write(str(accuracies[arff]["learningRate"]))
    results.write(",")
    if "momentum" in accuracies[arff].keys():
        results.write(str(accuracies[arff]["momentum"]))
    results.write("\n")
    
results.write(",,"+str(totalTestAccuracy/numTestAccuracies)+"\n")

results.close()
print "Average test accuracy",totalTestAccuracy/numTestAccuracies

