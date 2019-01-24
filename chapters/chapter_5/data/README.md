Getting Data!
==

We have tried to make getting the data as simple as possible.  Run the shell script (`get-all-data.sh`) provided in this folder to create the subfolder structure and download the data files needed for the notebooks. The files are hosted on google drive. 

There is one data file that we do not re-host: the GloVe word embeddings.  Please download from the stanford website: http://nlp.stanford.edu/data/glove.6B.zip.  Then, unzip and put the 100d version into a subfolder named `glove` to result in the following file path: `data/glove/glove.6B.100d.txt`