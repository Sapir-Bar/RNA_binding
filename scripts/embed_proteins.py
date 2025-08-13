import torch
import esm

# -------- settings --------
input_file  = "test_RBPs.txt"          # one protein sequence per line
output_file = "embeddings_test.pt"          # saves {"ids": [...], "embeddings": tensor}
model_name  = "esm2_t33_650M_UR50D"    # try: "esm2_t30_150M_UR50D" if still OOM
batch_size  = 2                        # reduce to 1 if needed
# --------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

# pick mixed-precision dtype (GPU only)
use_amp = device == "cuda"
amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else (torch.float16 if use_amp else torch.float32)

# load sequences (assume valid AA, length < 1022)
with open(input_file, "r") as f:
    seqs = [line.strip() for line in f if line.strip()]
data = [(f"seq{i+1}", s) for i, s in enumerate(seqs)]

all_ids, all_vecs = [], []
num_layers = model.num_layers  # last layer index

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

with torch.no_grad():
    for batch in chunks(data, batch_size):
        labels, strs, toks = batch_converter(batch)
        toks = toks.to(device)

        # mixed precision forward (GPU)
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else torch.cpu.amp.autocast(enabled=False)
        with ctx:
            out = model(toks, repr_layers=[num_layers], return_contacts=False)
            reps = out["representations"][num_layers]  # [B, L, D]

        for (seq_id, _seq), rep in zip(batch, reps):
            vec = rep[1:-1].mean(0).float().cpu()  # pool over tokens, drop [CLS]/[EOS]
            all_ids.append(seq_id)
            all_vecs.append(vec)

        # free GPU memory between batches
        del toks, out, reps
        if device == "cuda":
            torch.cuda.empty_cache()

emb = torch.stack(all_vecs)  # [N, D]
torch.save({"ids": all_ids, "embeddings": emb}, output_file)
print(f"Saved {len(all_ids)} embeddings to {output_file} with shape {tuple(emb.shape)}")
