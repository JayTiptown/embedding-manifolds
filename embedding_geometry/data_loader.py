import torch
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
from datasets import load_dataset


class Vocabulary:
    """Simple vocabulary for word embeddings."""
    def __init__(self, min_freq=5):
        self.word2idx = {}
        self.idx2word = {}
        self.min_freq = min_freq
        self.word_freq = Counter()
    
    def build(self, texts):
        """Build vocabulary from texts."""
        for text in texts:
            self.word_freq.update(text.lower().split())
        
        self.word2idx = {'<UNK>': 0}
        self.idx2word = {0: '<UNK>'}
        
        idx = 1
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def __len__(self):
        return len(self.word2idx)
    
    def get_idx(self, word):
        return self.word2idx.get(word.lower(), 0)


class WordSimDataset:
    """WordSim-353 dataset for similarity evaluation."""
    def __init__(self):
        self.pairs = []
        self.load_data()
    
    def load_data(self):
        """Load WordSim-353 dataset from Hugging Face."""
        dataset = load_dataset("almogtavor/WordSim353", split="train")
        
        for item in dataset:
            word1 = str(item['Word 1']).lower()
            word2 = str(item['Word 2']).lower()
            score = float(item['Human (Mean)'])
            self.pairs.append((word1, word2, score))
    
    def __len__(self):
        return len(self.pairs)
    
    def get_vocab_words(self):
        """Get unique words from dataset."""
        words = set()
        for w1, w2, _ in self.pairs:
            words.add(w1)
            words.add(w2)
        return list(words)


class GoogleAnalogyDataset:
    """Google Analogies dataset for analogy evaluation."""
    def __init__(self):
        self.analogies = []
        self.load_data()
    
    def load_data(self):
        """Load Google Analogies dataset from Hugging Face."""
        dataset = load_dataset("almogtavor/google-analogy-dataset", split="train")
        
        for item in dataset:
            category = str(item['Subject'])
            self.analogies.append({
                'category': category,
                'a': str(item['Word1']).lower(),
                'b': str(item['Word2']).lower(),
                'c': str(item['Word3']).lower(),
                'd': str(item['Word4']).lower()
            })
    
    def __len__(self):
        return len(self.analogies)
    
    def get_vocab_words(self):
        """Get unique words from dataset."""
        words = set()
        for analogy in self.analogies:
            words.update([analogy['a'], analogy['b'], analogy['c'], analogy['d']])
        return list(words)


class SkipGramDataset(Dataset):
    """Skip-gram training dataset."""
    def __init__(self, text, vocab, window_size=5):
        self.vocab = vocab
        self.window_size = window_size
        self.pairs = self._generate_pairs(text)
    
    def _generate_pairs(self, text):
        """Generate (center, context) pairs."""
        words = text.lower().split()
        word_ids = [self.vocab.get_idx(w) for w in words]
        
        pairs = []
        for i, center in enumerate(word_ids):
            if center == 0:
                continue
            
            window = np.random.randint(1, self.window_size + 1)
            for j in range(max(0, i - window), min(len(word_ids), i + window + 1)):
                if i != j and word_ids[j] != 0:
                    pairs.append((center, word_ids[j]))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


def load_text_corpus(dataset_name='wikitext', config='wikitext-2-raw-v1', split='train', max_chars=None):
    """
    Load text corpus from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset (default: 'wikitext')
        config: Dataset configuration (default: 'wikitext-2-raw-v1')
        split: Dataset split to load (default: 'train')
        max_chars: Maximum number of characters to use (None for all)
    
    Returns:
        text: Combined text corpus
    """
    dataset = load_dataset(dataset_name, config, split=split)
    
    texts = []
    for item in dataset:
        if 'text' in item:
            texts.append(item['text'])
    
    text = ' '.join(texts)
    
    if max_chars is not None:
        text = text[:max_chars]
    
    return text

