# Chaper 4

## Simple Example: XOR

We include a notebook for setting up the XOR problem using the two-dimensional toy data from Chapter 3. The notebook will show the Perceptron (which is technically a 1-layer MLP), a 2-layer MLP, and a 3-layer MLP. Various plots are also shown. 

### Notebooks

- [4_mlp_2d_points/2Dimensional_Perceptron_MLP](4_mlp_2d_points/2Dimensional_Perceptron_MLP.ipynb)

### Corresponding examples in the book

No actual code examples are shown in the book for the XOR problem, but the plots in Figure 4-3, Figure 4-4, and Figure 4-5 were derived from this notebook. 

## In Text Examples

### Notebooks

- [Chapter-4-In-Text-Examples](Chapter-4-In-Text-Examples.ipynb)
 
### Corresponding examples in the book

- Example 4-1. Multilayer perceptron using PyTorch
- Example 4-2. An example instantiation of an MLP
- Example 4-3. Testing the MLP with random inputs
- Example 4-4. Producing probabilistic outputs with a multilayer perceptron classifier (notice the apply_softmax = True option)
- Example 4-13. MLP with dropout
- Example 4-14. Artificial data and using a Conv1d class
- Example 4-15. The iterative application of convolutions to data
- Example 4-16. Two additional methods for reducing to feature vectors
- Example 4-22. Using a Conv1D layer with batch normalization

## Example: Surname Classification with an MLP

### Notebooks

- (for dataset preprocessing) [munging_surname_dataset](4_2_mlp_surnames/munging_surname_dataset.ipynb)
- [4_2_Classifying_Surnames_with_an_MLP](4_2_mlp_surnames/4_2_Classifying_Surnames_with_an_MLP.ipynb)
 
### Corresponding examples in the book


- Example 4-5. Implementing `SurnameDataset.__getitem__()`
- Example 4-6. Implementing SurnameVectorizer
- Example 4-7. The SurnameClassifier using an MLP
- Example 4-8. Hyperparameters and program options for the MLP-based Yelp review classifier
- Example 4-9. Instantiating the dataset, model, loss, and optimizer
- Example 4-10. A snippet of the training loop
- Example 4-11. Inference using an existing model (classifier): Predicting the nationality given a name
- Example 4-12. Predicting the top-k nationalities

## Example: Classifying Surnames by Using a CNN

### Notebooks

- (for dataset preprocessing) [munging_surname_dataset](4_4_cnn_surnames/munging_surname_dataset.ipynb)
- [4_4_Classifying_Surnames_with_a_CNN](4_4_cnn_surnames/4_4_Classifying_Surnames_with_a_CNN.ipynb)
 
### Corresponding examples in the book


- Example 4-17. SurnameDataset modified for passing the maximum surname length
- Example 4-18. Implementing the SurnameVectorizer for CNNs
- Example 4-19. The CNN-based SurnameClassifier
- Example 4-20. Input arguments to the CNN surname classifier
- Example 4-21. Using the trained model to make predictions
