import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler

# from transformers import get_cosine_schedule_with_warmup

# Special tokens
PAD_ID   = 768
START_ID = 769
END_ID   = 770
PAD_SRC_ID   = 27495

# ─── Dataset + DataLoader ──────────────────────────────────────────────────────
class CodebookDataset(Dataset):
    def __init__(self, source_dir, code_dir):
        self.source_files = sorted(os.listdir(source_dir))
        self.code_files   = sorted(os.listdir(code_dir))
        self.source_data, self.code_data = [], []

        for src_f, tgt_f in zip(self.source_files, self.code_files):
            src_batch = torch.load(os.path.join(source_dir, src_f), map_location="cpu")
            tgt_batch = torch.load(os.path.join(code_dir,   tgt_f), map_location="cpu")
            self.source_data.extend(src_batch)
            self.code_data.extend(tgt_batch)
            # break

        assert len(self.source_data) == len(self.code_data)

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, idx):
        src = torch.tensor(self.source_data[idx], dtype=torch.long)
        tgt_raw = self.code_data[idx].long()
        tgt = torch.cat([
            torch.tensor([START_ID], dtype=torch.long),
            tgt_raw,
            torch.tensor([END_ID],   dtype=torch.long)
        ])
        return src, tgt

def collate_fn(batch):
    src_seq, tgt_seq = zip(*batch)
    src_padded = pad_sequence(src_seq, batch_first=True, padding_value=PAD_SRC_ID)
    tgt_padded = pad_sequence(tgt_seq, batch_first=True, padding_value=PAD_ID)
    return src_padded, tgt_padded

# ─── Model Definition ───────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=800):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size=27500, tgt_vocab_size=771,
                 d_model=768, nhead=8, num_layers=8,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.pad_token_id = PAD_ID

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_SRC_ID)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc     = PositionalEncoding(d_model, dropout, max_len=800)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt_input):
        # src: (B, S), tgt_input: (B, T)
        src_pad_mask = src == PAD_SRC_ID       # (B, S) bool
        tgt_pad_mask = tgt_input == PAD_ID # (B, T) bool
        causal_mask  = torch.triu(torch.ones((tgt_input.size(1),)*2, dtype=torch.bool, device=src.device), diagonal=1)

        src_emb = self.pos_enc(self.src_tok_emb(src))
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt_input))

        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        out = self.decoder(tgt_emb, memory,
                           tgt_mask=causal_mask,
                           tgt_key_padding_mask=tgt_pad_mask,
                           memory_key_padding_mask=src_pad_mask)
        return self.output_linear(out)  # (B, T, V)

# ─── Training Loop ─────────────────────────────────────────────────────────────
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            logits = model(src, tgt_input)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_target.reshape(-1))
            num_tokens = (tgt_target != PAD_ID).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return avg_loss

def train():
    # Paths
    source_dir = "/home/naren-root/Documents/SIP/Data/Train/Source_Processed/"
    code_dir   = "/home/naren-root/Documents/SIP/Data/Train/Codes_Proper/"
    checkpoint_dir = "/home/naren-root/Documents/SIP/Custom-Impl/checkpoints_transformer"
    log_file       = "/home/naren-root/Documents/SIP/Custom-Impl/transformer_pose.log"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Logger
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Data
    dataset = CodebookDataset(source_dir, code_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True,
                            collate_fn=collate_fn, pin_memory=True, num_workers=2)

    # Add dev set
    dev_source_dir = "/home/naren-root/Documents/SIP/Data/Dev/Source_Processed/"
    dev_code_dir   = "/home/naren-root/Documents/SIP/Data/Dev/Codes_Proper/"
    dev_dataset = CodebookDataset(dev_source_dir, dev_code_dir)
    dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False,
                                collate_fn=collate_fn, pin_memory=True, num_workers=2)

    # Model & Optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerSeq2Seq(nhead=4, num_layers=4).to(device)
    # data = torch.load("/home/naren-root/Documents/SIP/Custom-Impl/checkpoints_transformer/transformer_model_overfit.pt")
    # model.load_state_dict(data["model_state"])
    # logging.info(f"Loaded model checkpoint with avg loss : {data['avg_loss']}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    class_weights = torch.load("class_weights.pt").to("cuda")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    scaler = GradScaler()

    # Train
    EPOCHS = 350 #200
    best_loss = 1000
    best_dev_loss = float('inf')
    #start from 19
    num_training_steps = EPOCHS * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)
    
    # scheduler = get_cosine_schedule_with_warmup(
    # optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    # )
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        token_count = 0

        for batch_idx, (src, tgt) in enumerate(dataloader, 1):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            # shift for teacher forcing
            tgt_input  = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            with autocast():
                logits = model(src, tgt_input)                      # (B, T, V)
                loss   = criterion(logits.view(-1, logits.size(-1)),
                                   tgt_target.reshape(-1))
            if batch_idx == 1 or batch_idx % 100 == 0:
                pred_tokens = logits.argmax(-1)  # (B, T)
                # print("Sample predictions vs targets (first 2 in batch):")
                for i in range(min(2, pred_tokens.size(0))):
                    print("Src:", src[i, :].tolist())
                    print("Pred:", pred_tokens[i, :30].tolist())
                    print("Logits:", logits[i, :])
                    print("Tgt :", tgt[i, :30].tolist())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()  # <= important

            # accumulate for logging
            num_tokens = (tgt_target != PAD_ID).sum().item()
            epoch_loss += loss.item() * num_tokens
            token_count += num_tokens

            logging.info(f"[Epoch {epoch}][Batch {batch_idx}] loss={loss.item():.4f}")
            # print(f"Epoch {epoch} Batch {batch_idx}: Loss={loss.item():.4f}")

        avg_loss = epoch_loss / token_count
        logging.info(f"Epoch {epoch} complete. Avg loss per token: {avg_loss:.6f}")
        # print(f"Epoch {epoch} finished. Avg loss: {avg_loss:.6f}")

        # Evaluate on dev set
        dev_loss = evaluate(model, dev_dataloader, device, criterion)
        logging.info(f"Epoch {epoch} dev set avg loss per token: {dev_loss:.6f}")

        # save checkpoint if best dev loss
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'avg_loss': avg_loss,
            'dev_loss': dev_loss
        }
        path = os.path.join(checkpoint_dir, f"transformer_model_check.pt")
        if dev_loss <= best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(ckpt, path)
            logging.info(f"Best dev loss checkpoint saved to {path}")

    logging.info("Training complete!")

if __name__ == "__main__":
    train()