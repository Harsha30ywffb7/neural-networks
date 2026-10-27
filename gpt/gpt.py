import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


class Softmax:
    def __init__(self):
        self.softmax_out = None

    def forward(self, inputs):
        # to avoid softmax instability due to shift invariant remove the max number.
        max_logit = np.max(inputs, axis=-1, keepdims=True)

        softmax_out = np.exp(inputs - max_logit)/np.sum(np.exp(inputs-max_logit), axis=-1, keepdims=True)
        self.softmax_out = softmax_out

        return softmax_out
