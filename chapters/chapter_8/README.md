# Chapter 8

## In Text Examples

### Notebooks

- [8_PackedSequence_example](8_PackedSequence_example.ipynb)

### Corresponding examples in the book

- Example 8-6. A simple demonstration of packed_padded_sequences and pad_packed_sequences


##  Example: Neural Machine Translation

### Notebooks

- (for dataset preprocessing) [8_5_nmt_munging](8_5_NMT/8_5_nmt_munging.ipynb)
- [8_5_NMT_No_Sampling](8_5_NMT/8_5_NMT_No_Sampling.ipynb)

### Corresponding examples in the book

- Example 8-1. Constructing the NMTVectorizer
- Example 8-2. The vectorization functions in the NMTVectorizer
- Example 8-3. Generating minibatches for the NMT example
- Example 8-4. The NMTModel encapsulates and coordinates the encoder and decoder in a single forward() method
- Example 8-5. The encoder embeds the source words and extracts features with a biGRU
- Example 8-7. The NMTDecoder constructs a target sentence from the encoded source sentence
- Example 8-8. Attention mechanism that does element-wise multiplication and summing more explicitly

## NMT using sampling

### Notebooks

- [8_5_NMT_scheduled_sampling](8_5_NMT/8_5_NMT_scheduled_sampling.ipynb)

### Corresponding examples in the book

- Example 8-9. The decoder with a sampling procedure (in bold) built into the forward pass