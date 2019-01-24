# Chapter 3

In Chapter 3, you are introduced to the supervised training routine and walked through two examples.  The first example uses toy two-dimensional data and a Perceptron in binary classification task.  The second example uses the Yelp review dataset and a Perceptron in a binary classification task. 

Important note:  we make available the raw dataset and the "light" version in the data download script you should have ran.  If you want to use the "full" version, you should (1) run the "full" munging notebook and (2) change the path in the `args` object to point to the "full" version. 

## In Text Examples

### Notebooks

- [Chapter-3-In-Text-Examples](Chapter-3-In-Text-Examples.ipynb)

### Corresponding examples in the book

- Example 3-1. Implementing a perceptron using PyTorch
- Example 3-2. Sigmoid activation
- Example 3-3. Tanh activation
- Example 3-4. ReLU activation
- Example 3-5. PReLU activation
- Example 3-6. Softmax activation
- Example 3-7. MSE loss
- Example 3-8. Cross-entropy loss
- Example 3-9. Binary cross-entropy loss

## Diving Deep into Supervised Training

### Notebooks

- [Chapter-3-Diving-Deep-into-Supervised-Training](Chapter-3-Diving-Deep-into-Supervised-Training.ipynb)

### Corresponding examples in the book

- Example 3-10. Instantiating the Adam optimizer
- Example 3-11. A supervised training loop for a perceptron and binary classification 

## Example: Classifying Sentiment of Restaurant Reviews

### Notebooks

- (for dataset preprocessing) ["light"](3_5_yelp_dataset_preprocessing_LITE.ipynb) or ["full"](3_5_yelp_dataset_preprocessing_FULL.ipynb) 
- [3_5_Classifying_Yelp_Review_Sentiment](3_5_Classifying_Yelp_Review_Sentiment.ipynb)

### Corresponding examples in the book

- Example 3-12. Creating training, validation, and testing splits
- Example 3-13. Minimally cleaning the data
- Example 3-14. A PyTorch Dataset class for the Yelp Review dataset
- Example 3-15. The Vocabulary class maintains token to integer mapping needed for the rest of the machine learning pipeline
- Example 3-16. The Vectorizer class converts text to numeric vectors
- Example 3-17. Generating minibatches from a dataset
- Example 3-18. A perceptron classifier for classifying Yelp reviews
- Example 3-19. Hyperparameters and program options for the perceptron-based Yelp review classifier
- Example 3-20. Instantiating the dataset, model, loss, optimizer, and training state
- Example 3-21. A bare-bones training loop
- Example 3-22. Test set evaluation
- Example 3-23. Printing the prediction for a sample review
- Example 3-24. Inspecting what the classifier learned
