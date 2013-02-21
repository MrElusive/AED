AED
===

Automatic Emotion Detector

Setting up the libraries:


[Feature Extractor]

Built in Python. You'll need the following library:

-pyopencv (http://opencv.willowgarage.com/wiki/InstallGuide_Linux)
You can try to follow the install guide in the link above, or you can just use a Linux package manager to install the following packages: python-opencv and libopencv-core2.3 

You can also install the following library, but it's optional for now:

-PIL (pip install PIL)
The Python Imaging Library is useful for lots of simple image editing and filtering operations. However, we may end up just using pyopencv to do everything for us.

To install pip:	sudo apt-get install python-pip


[Machine Learner]
Built in C++.
We shouldn't need any special libraries for this component, besides the ML framework from class.  We should use Python to extract features because it's relatively easy to experiment and program with, but we should use C++ for the learner because it'll be faster.
