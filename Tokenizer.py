class tokenizer:

    def __init__(self) -> None:
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}


    def train(self, text, vocab_size, forget = True):
        '''
        Train the tokenizer using given text till we reach given vocab size
        '''
        if forget == True:
            self.clear()

        ids = list(text.encode('utf-8'))    # encode text as byte
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # print(f"mergeing {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx

        # update the vocab map
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    
    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens)>=2:
            stats = self.get_stats(tokens)
            pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]

            tokens = self.merge(tokens, pair, idx)
        return tokens
    
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors='replace')

        return text



    def get_stats(self, ids):
        '''
        Count the frequency of the byte pairs for byte pairing procedure
        '''
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        '''
        Given pair and text, merge text based on given pairs
        '''
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids)- 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def clear(self):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}