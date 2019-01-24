# Chapter 7 

## In Text Examples

- Example 7-12. Applying gradient clipping in PyTorch
	+ There will not be a dedicated notebook for this example
	+ instead, please see the following code snippet:

```
# after you call backward on a loss scalar:
loss.backward()
# you can then clip the gradient. see the documentation for more information
torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
```

## Example: A Character RNN for Generating Surnames

### Notebooks

- (for dataset preprocessing) [7_3_Munging_Surname_Dataset](7_3_surname_generation/7_3_Munging_Surname_Dataset.ipynb)
- [7_3_Model1_Unconditioned_Surname_Generation](7_3_surname_generation/7_3_Model1_Unconditioned_Surname_Generation.ipynb) 
- [7_3_Model2_Conditioned_Surname_Generation](7_3_surname_generation/7_3_Model2_Conditioned_Surname_Generation.ipynb)

### Corresponding examples in the book


- Example 7-1. The SurnameDataset.__getitem__() method for a sequence prediction task
- Example 7-2. The code for SurnameVectorizer.vectorize() in a sequence prediction task
- Example 7-3. The unconditioned surname generation model
- Example 7-4. The conditioned surname generation model
- Example 7-5. Handling three-dimensional tensors and sequence-wide loss computations
- Example 7-6. Hyperparameters for surname generation
- Example 7-7. Sampling from the unconditioned generation model
- Example 7-8. Mapping sampled indices to surname strings
- Example 7-9. Sampling from the unconditioned model
- Example 7-10. Sampling from a sequence model
- Example 7-11. Sampling from the conditioned SurnameGenerationModel (not all outputs are shown)