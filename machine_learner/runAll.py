import os
cmd = 'ls -l /usr/bin'
os.system('ls ../feature_extractor/arffoutput > outputs/arffFiles.txt')

arffFilesFile = open("outputs/arffFiles.txt",'r')
arffFiles = arffFilesFile.readlines()
#remove newline characters
arffFiles = map((lambda x:x[0:len(x)-1]),arffFiles)
numFiles = len(arffFiles)
fileNum = 1
for arff in arffFiles:
    print "Running file",fileNum,"of",numFiles
    dimensions = (int)(arff[arff.rfind('-')+1:arff.rfind('.')-1])
    print "Running",arff,"with",dimensions,"dimensions"
    
    command = "./bin/MLSystemManager -L backprop -A ../feature_extractor/arffoutput/"
    command += arff
    command += " -l 1 -H " + str(dimensions*2)
    command += " -E random 0.75 "
    command += " > outputs/"+arff[0:arff.rfind('.')]+"_output.txt"
    os.system(command)
    fileNum += 1

