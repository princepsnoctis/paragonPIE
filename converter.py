class Converter:
    def __init__(self, symbols, special_symbols=[]):
        self.unknown_symbol = "<UNK>"

        self.symbols = [self.unknown_symbol] + special_symbols + symbols

        self.symbol2index = {s:i for i,s in enumerate(self.symbols)}
        self.index2symbol = {i:s for i,s in enumerate(self.symbols)}

    def __len__(self):
        return len(self.symbols)

    def encode(self, symbol):
        return self.symbol2index.get(symbol, self.symbol2index[self.unknown_symbol])

    def decode(self, index):
        return self.index2symbol.get(index, self.unknown_symbol)

    def encode_seq(self, symbols):
        return [self.encode(symbol) for symbol in symbols]

    def decode_seq(self, indices):
        return [self.decode(index) for index in indices]

    def __getitem__(self, symbol):
        return self.encode(symbol)