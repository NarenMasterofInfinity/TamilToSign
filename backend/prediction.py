import torch
from transformer_pose import TransformerSeq2Seq, PAD_ID, PAD_SRC_ID, START_ID, END_ID

@torch.no_grad()
def generate(model, src_seq, max_len=100):
    model.eval()
    device = next(model.parameters()).device

    # Prepare input
    src = torch.tensor(src_seq, dtype=torch.long).unsqueeze(0).to(device)  # (1, S)
    src_mask = src == PAD_SRC_ID

    # Encode source
    src_emb = model.pos_enc(model.src_tok_emb(src))
    memory = model.encoder(src_emb, src_key_padding_mask=src_mask)

    # Initialize decoder input with START_ID
    tgt_tokens = [START_ID]
    
    for _ in range(max_len):
        tgt_input = torch.tensor(tgt_tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, T)
        tgt_mask = torch.triu(torch.ones((tgt_input.size(1), tgt_input.size(1)), device=device, dtype=torch.bool), diagonal=1)

        tgt_emb = model.pos_enc(model.tgt_tok_emb(tgt_input))
        out = model.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=src_mask
        )
        logits = model.output_linear(out)  # (1, T, V)
        next_token = logits[0, -1].argmax().item()
        if next_token == END_ID:
            break
        tgt_tokens.append(next_token)

        

    return tgt_tokens[1:]