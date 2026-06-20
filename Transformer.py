import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Scaled Dot-Product Attention

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
            attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


# 2. Multi-Head Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
       super().__init__()
       assert d_model % num_heads == 0

       self.d_model = d_model
       self.num_heads = num_heads
       self.d_k = d_model // num_heads

       self.W_q = nn.Linear(d_model, d_model)
       self.W_k = nn.Linear(d_model, d_model)
       self.W_v = nn.Linear(d_model, d_model)
       self.W_o = nn.Linear(d_model, d_model)

       self.dropout = nn.Dropout(dropout)
       self.attn_weights = None

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1 ,2)

    # Need to try forward, backward and Bi-direction later
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        x, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(x)


# 3. Position-wise Feed-Forward Network

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# 4. Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
       super().__init__()
       self.dropout = nn.Dropout(dropout)

       pe = torch.zeros(max_len, d_model)
       position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
       div_term = torch.exp(
           torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
       )

       pe[:, 0::2] = torch.sin(position * div_term)
       pe[:, 1::2] = torch.cos(position * div_term)

       pe = pe.unsqueeze(0)
       self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 5. Add & Norm

class AddAndNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))


# 6. Encoder Layer

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)

    def forward(self, x, src_mask=None):
        x = self.add_norm1(x, self.self_attn(x, x, x, src_mask))
        x = self.add_norm2(x, self.ffn(x))
        return x


# 7. Decoder Layer

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm1  = AddAndNorm(d_model, dropout)
        self.add_norm2  = AddAndNorm(d_model, dropout)
        self.add_norm3  = AddAndNorm(d_model, dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
       x = self.add_norm1(x, self.self_attn(x, x, x, tgt_mask))
       x = self.add_norm2(x, self.cross_attn(x, enc_output, enc_output, src_mask))
       x = self.add_norm3(x, self.ffn(x))
       return x


# 8. Completed Encoder/Decoder(n layers)

class Encoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, num_layers):
       super().__init__()
       self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
       self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


# 9. Completed Transformer

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            dropout=0.1,
            max_seq_len=5000,
    ):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.encoder = Encoder(enc_layer, num_encoder_layers)
        self.decoder = Decoder(dec_layer, num_decoder_layers)

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src,pad_idx=0):
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt, pad_idx=0):
        tgt_len = tgt.size(1)

        subsequent_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).bool()

        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

        return pad_mask & subsequent_mask

    def encode(self, src, src_mask):
        x = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.decoder(x, enc_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        src_mask = self.make_src_mask(src, src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, tgt_pad_idx)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)

        logits = self.output_projection(dec_output)
        return logits


# 10. Test Code

if __name__ == "__main__":
    print("=" * 60)
    print("Transformer (Attention is all you need) _Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    BATCH_SIZE   =2
    SRC_SEQ_LEN  = 10
    TGT_SEQ_LEN  = 8
    SRC_VOCAB    = 1000
    TGT_VOCAB    = 1000
    D_MODEL      = 512
    NUM_HEADS    = 8
    N_LAYERS     = 6
    D_FF         = 2048
    DROPOUT      = 0.1
    PAD_IDX      = 0

    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=N_LAYERS,
        num_decoder_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Numbers of training parameters: {total_params:,}")
    print(f"(Original numbers of parameters is 65M)\n")

    src = torch.randint(1, SRC_VOCAB, (BATCH_SIZE, SRC_SEQ_LEN)).to(device)
    tgt = torch.randint(1, TGT_VOCAB, (BATCH_SIZE, TGT_SEQ_LEN)).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(src, tgt, src_pad_idx=PAD_IDX, tgt_pad_idx=PAD_IDX)

        print(f"Input shape of src: {src.shape} -> (batch={BATCH_SIZE}, src_len={SRC_SEQ_LEN})")
        print(f"Input shape of tgt: {tgt.shape} -> (batch={BATCH_SIZE}, tgt_len={TGT_SEQ_LEN})")
        print(f"Input shape of logits: {logits.shape} -> (batch, tgt_len, tgt_size)")
        print(f"\n forward translation success!")

    print("\nExample for single step")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    tgt_input = tgt[:, :-1]
    tgt_labels = tgt[:, 1:]

    logits = model(src, tgt_input, src_pad_idx=PAD_IDX, tgt_pad_idx=PAD_IDX)
    loss = criterion(logits.reshape(-1, TGT_VOCAB), tgt_labels.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print("\n backward + parameter upgrade successfully!")