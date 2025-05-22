import pandas as pd
import nltk
from collections import defaultdict
import math
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class NGramModel:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()

    def train(self, texts):
        for text in texts:
            tokens = ["<s>"] * (self.n - 1) + word_tokenize(text.lower()) + ["</s>"]
            self.vocab.update(tokens)
            for ngram in ngrams(tokens, self.n):
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
        self.vocab_size = len(self.vocab)

    def get_prob(self, ngram):
        context = ngram[:-1]
        count_ngram = self.ngram_counts[ngram]
        count_context = self.context_counts[context]
        prob = (count_ngram + self.k) / (count_context + self.k * self.vocab_size)
        return prob

    def score(self, text):
        tokens = ["<s>"] * (self.n - 1) + word_tokenize(text.lower()) + ["</s>"]
        score = 0.0
        for ngram in ngrams(tokens, self.n):
            prob = self.get_prob(ngram)
            score += math.log(prob)
        return score

def classify(model_human, model_ai, text):
    score_human = model_human.score(text)
    score_ai = model_ai.score(text)
    return 0 if score_human > score_ai else 1

# Optional: Run a demo ONLY when called directly
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df = pd.read_csv('data.csv')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    n_values = [2, 3]
    k_values = [0.1, 0.5, 1.0]
    best_acc = 0
    best_n = None
    best_k = None

    for n in n_values:
        for k in k_values:
            model_human = NGramModel(n=n, k=k)
            model_ai = NGramModel(n=n, k=k)

            human_texts = train_df[train_df['generated'] == 0]['text']
            ai_texts = train_df[train_df['generated'] == 1]['text']

            model_human.train(human_texts)
            model_ai.train(ai_texts)

            preds = val_df['text'].apply(lambda x: classify(model_human, model_ai, x))
            acc = accuracy_score(val_df['generated'], preds)

            print(f'n={n}, k={k}, Validation Accuracy: {acc:.4f}')

            if acc > best_acc:
                best_acc = acc
                best_n = n
                best_k = k

    print(f'\nBest Hyperparameters: n={best_n}, k={best_k}, Accuracy={best_acc:.4f}')