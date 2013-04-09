import os

os.system("ls outputs/*output.txt > outputs/outputList.txt")

outputs = open("outputs/outputList.txt").readlines()
outputs = map((lambda x:x[0:len(x)-1]),outputs)

#This is a dictionary of dictionaries
accuracies = {}
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
    accuracies[o] = setAccuracies
        
    f.close()
results = open("outputs/results.csv",'w')
results.write("Arff File, Training Accuracy, Test Accuracy\n")
for arff in accuracies.keys():
    print "Writing data for",arff
    print accuracies[arff]
    results.write(arff+",")
    if "train" in accuracies[arff].keys():
        results.write(str(accuracies[arff]["train"]))
    results.write(",")
    if "test" in accuracies[arff].keys():
        results.write(str(accuracies[arff]["test"]))
    results.write("\n")

results.close()


