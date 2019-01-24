# Chapter 5

## In Text Examples

### Notebooks

- [5_1_Pretrained_Embeddings](5_1_Pretrained_Embeddings.ipynb)

### Corresponding examples in the book

- Example 5-1. Using pretrained word embeddings
- Example 5-2. The analogy task using word embeddings
- Example 5-3. Word embeddings encode many linguistics relationships, as illustrated using the SAT analogy task
- Example 5-4. An example illustrating the danger of using cooccurrences to encode meaning—sometimes they do not!
- Example 5-5. Watch out for protected attributes such as gender encoded in word embeddings. This can introduce unwanted biases in downstream models.
- Example 5-6. Cultural gender bias encoded in vector analogy

## Example: Learning the Continuous Bag of Words Embeddings

### Notebooks

- (for dataset preprocessing) [5_2_munging_frankenstein](5_2_CBOW/5_2_munging_frankenstein.ipynb)
- [5_2_Continuous_Bag_of_Words_CBOW](5_2_CBOW/5_2_Continuous_Bag_of_Words_CBOW.ipynb)

### Corresponding examples in the book

- Example 5-7. Constructing a dataset class for the CBOW task
- Example 5-8. A Vectorizer for the CBOW data
- Example 5-9. The CBOWClassifier model
- Example 5-10. Arguments to the CBOW training script

## Example: Transfer Learning Using Pretrained Embeddings for Document Classification


### Notebooks

- (for dataset preprocessing) [5_3_Munging_AG_News](5_3_doc_classification/5_3_Munging_AG_News.ipynb)
- [5_3_Document_Classification_with_CNN](5_3_doc_classification/5_3_Document_Classification_with_CNN.ipynb)


### Corresponding examples in the book

- Example 5-11. The `NewsDataset.__getitem__()` method
- Example 5-12. Implementing a Vectorizer for the AG News dataset
- Example 5-13. Selecting a subset of the word embeddings based on the vocabulary
- Example 5-14. Implementing the NewsClassifier
- Example 5-15. Arguments to the CNN NewsClassifier using pretrained embeddings
- Example 5-16. Predicting with the trained model