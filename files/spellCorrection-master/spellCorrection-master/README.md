### To RUN
uncomment the lines:


```python
datasetFile = sys.argv[1];			# line 11 
outFile = sys.argv[2]; 				 # line 12
```
so the program gets the command line arguments, otherwise program will look for "dataset.txt" for in the working directory and the "output.txt " will be in the working directory too.
>  argv[1] 

indicates the datasetFile.

> argv[2]

indicates the outputFile.

##### Default Test Data Size :
Default test data size is 200. You can change it in the line :
```python
testDataSize = 200 # default is 200

### Important Notes
1-when hiden markov model is created and initialized, hmm will use all the data. But
there is a memory leak in the viterbi alg. So, when i use all the data for testing, python3 kernel will throws memory error. I was not able to resolve the bug( i think it is because there is a circular reference in some of the loops and garbage collector of python3 was not able to resolve the problem). Change the testDataSize accordingly. 

2- initializing hmm could take severel mins(2-5).
