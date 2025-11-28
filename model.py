import torch

class Embedding(torch.nn.Module):
    def __init__(self, sym_len, max_seq_len, emb_dim=512):
        super().__init__()

        self.sym_len = sym_len
        self.max_seq_len = max_seq_len

        self.sym_embedding = torch.nn.Embedding(num_embeddings=sym_len,     embedding_dim=emb_dim)
        self.pos_embedding = torch.nn.Embedding(num_embeddings=max_seq_len, embedding_dim=emb_dim)

    def forward(self, x):
        B, L = x.shape

        pos = torch.arange(L, device=x.device).unsqueeze(0) # (1, L)

        sym_emb = self.sym_embedding(x)   # (B, L, D)
        pos_emb = self.pos_embedding(pos) # (1, L, D)

        x = sym_emb + pos_emb

        return x

class Encoder(torch.nn.Module):
    def __init__(self, emb_dim=512, num_heads=8, ff_dim=2048, dropout=0.1, num_layers=4):
        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True, # More stable training
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False, # For norm first
        )

        self.final_norm = torch.nn.LayerNorm(emb_dim)

    def forward(self, x, sample_mask):
        x = self.encoder(x, src_key_padding_mask=sample_mask)
        x = self.final_norm(x)

        return x

class Decoder(torch.nn.Module):
    def __init__(self, emb_dim=512, num_heads=8, ff_dim=2048, dropout=0.1, num_layers=4):
        super().__init__()

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = torch.nn.LayerNorm(emb_dim)

    def forward(self, decoder_input, encoder_output, name_mask, sample_mask):
        name_seq_len = decoder_input.size(1)

        causal_mask = torch.triu(torch.ones((name_seq_len, name_seq_len), dtype=torch.bool, device=decoder_input.device), diagonal=1)

        x = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=name_mask,
            memory_key_padding_mask=sample_mask
        )

        x = self.final_norm(x)

        return x

class SeqHead(torch.nn.Module):
    def __init__(self, sym_len, emb_dim=512):
        super().__init__()

        self.linear = torch.nn.Linear(in_features=emb_dim, out_features=sym_len)

    def forward(self, decoder_output):
        logits = self.linear(decoder_output)

        return logits

class CatHead(torch.nn.Module):
    def __init__(self, cat_len, emb_dim=512):
        super().__init__()

        self.linear = torch.nn.Linear(in_features=emb_dim, out_features=cat_len)

    def forward(self, encoder_output):
        first_token = encoder_output[:, 0, :]

        logits = self.linear(first_token)

        return logits

class RegHead(torch.nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, encoder_output):
        first_token = encoder_output[:, 0, :]

        pred = self.mlp(first_token)

        pred = torch.nn.functional.softplus(pred) # Ensure all values are non-negative for MSLE loss

        return pred

class BinHead(torch.nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, encoder_output):
        first_token = encoder_output[:, 0, :]

        logit = self.mlp(first_token)

        return logit

class MultiHead(torch.nn.Module):
    def __init__(self, sym_len, unit_cat_len, tax_cat_len, emb_dim=512):
        super().__init__()

        self.name_head             = SeqHead(sym_len=sym_len, emb_dim=emb_dim)

        self.unit_head             = CatHead(cat_len=unit_cat_len, emb_dim=emb_dim)
        self.tax_head              = CatHead(cat_len=tax_cat_len,  emb_dim=emb_dim)

        self.amount_head           = RegHead(emb_dim=emb_dim)
        self.quantity_head         = RegHead(emb_dim=emb_dim)
        self.price_head            = RegHead(emb_dim=emb_dim)
        self.total_head            = RegHead(emb_dim=emb_dim)

        self.amount_present_head   = BinHead(emb_dim=emb_dim)
        self.quantity_present_head = BinHead(emb_dim=emb_dim)
        self.price_present_head    = BinHead(emb_dim=emb_dim)
        self.total_present_head    = BinHead(emb_dim=emb_dim)

    def forward(self, encoder_output, decoder_output):
        return {
            "name_logits"           : self.name_head(decoder_output),

            "unit_logits"           : self.unit_head(encoder_output),
            "tax_logits"            : self.tax_head(encoder_output),

            "amount_pred"           : self.amount_head(encoder_output),
            "quantity_pred"         : self.quantity_head(encoder_output),
            "price_pred"            : self.price_head(encoder_output),
            "total_pred"            : self.total_head(encoder_output),

            "amount_present_logit"  : self.amount_present_head(encoder_output),
            "quantity_present_logit": self.quantity_present_head(encoder_output),
            "price_present_logit"   : self.price_present_head(encoder_output),
            "total_present_logit"   : self.total_present_head(encoder_output),
        }

class Model(torch.nn.Module):
    def __init__(
        self,
        sym_len,      # vocabulary size

        # These are for embeddings
        max_sam_len,  # max size of sample (input)
        max_nam_len,  # max size of name (output)

        # There are for heads
        unit_cat_len,
        tax_cat_len,

        # Hyperparams just for the model modules
        emb_dim=512,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        num_layers=4
    ):
        super().__init__()

        self.encoder_embedding = Embedding(
            sym_len=sym_len,
            max_seq_len=max_sam_len,
            emb_dim=emb_dim
        )

        self.decoder_embedding = Embedding(
            sym_len=sym_len,
            max_seq_len=max_nam_len, # max_sam_len and max_nam_len --- do we keep then different? does it matter?
            emb_dim=emb_dim
        )

        self.encoder = Encoder(
            emb_dim=emb_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_layers
        )

        self.decoder = Decoder(
            emb_dim=emb_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_layers
        )

        self.multihead = MultiHead(
            sym_len=sym_len,
            unit_cat_len=unit_cat_len,
            tax_cat_len=tax_cat_len,
            emb_dim=emb_dim
        )

    def forward(
            self,
            encoder_tokens,
            decoder_tokens,
            sample_mask,
            name_mask
    ):
        encoder_embedding = self.encoder_embedding(encoder_tokens)
        decoder_embedding = self.decoder_embedding(decoder_tokens)

        encoder_output = self.encoder(encoder_embedding, sample_mask)

        decoder_output = self.decoder(
            decoder_input=decoder_embedding,
            encoder_output=encoder_output,
            sample_mask=sample_mask,
            name_mask=name_mask,
        )

        outputs = self.multihead(
            encoder_output=encoder_output,
            decoder_output=decoder_output,
        )

        return outputs

    @torch.no_grad()
    def generate_name(
            self,
            encoder_input,
            bos_idx,
            eos_idx,
            max_nam_len,
    ):
        # Get encoding once
        encoder_input_embedding = self.encoder_embedding(encoder_input)

        sample_mask = torch.zeros_like(encoder_input).bool()

        encoder_output = self.encoder(encoder_input_embedding, sample_mask)

        # Generate sequence
        sequence = [bos_idx]

        for i in range(max_nam_len):
            decoder_input = torch.tensor(sequence).unsqueeze(0) # BOS

            decoder_input_embedding = self.decoder_embedding(decoder_input)

            name_mask = torch.zeros_like(decoder_input).bool()

            decoder_output = self.decoder(
                decoder_input=decoder_input_embedding,
                encoder_output=encoder_output,
                name_mask=name_mask,
                sample_mask=sample_mask,
            )

            logits = self.multihead.name_head(decoder_output)

            last_token_logits = logits[:, -1, :]

            last_token_probs = torch.nn.functional.softmax(last_token_logits, dim=-1)

            next_token = last_token_probs.argmax(dim=-1).item()

            sequence.append(next_token)

            if next_token == eos_idx:
                break

        return sequence