"""GeneFormer 2024+ 向后兼容"""
import os
import pickle

# Load token dictionary from the installed geneformer package
try:
    from geneformer import TOKEN_DICTIONARY_FILE
    with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
        token_dictionary = pickle.load(f)
    vocab_size = len(token_dictionary)
except Exception as e:
    raise ImportError(
        f"Failed to load geneformer token dictionary: {e}\n"
        "Make sure geneformer is properly installed with dictionary files."
    )
