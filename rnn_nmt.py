import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
import math
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import sacrebleu
from tqdm import tqdm

# ==========================================
# 0. 全局设置与工具
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. 数据集 (Robust Version)
# ==========================================
class TranslationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=64):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        # 自动匹配可能的键名
                        zn_key = next((k for k in obj.keys() if any(x in k.lower() for x in ['zn', 'zh', 'cn', 'source', 'src'])), None)
                        en_key = next((k for k in obj.keys() if any(x in k.lower() for x in ['en', 'eng', 'target', 'tgt'])), None)
                        
                        if zn_key and en_key and obj[zn_key] and obj[en_key]:
                            self.data.append({"src": str(obj[zn_key]).strip(), "tgt": str(obj[en_key]).strip()})
                    except json.JSONDecodeError: continue
        else:
            print(f"[Error] File not found: {file_path}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item['src']
        tgt_text = item['tgt']
        
        src_enc = self.tokenizer(src_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        tgt_enc = self.tokenizer(text_target=tgt_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        
        return {
            'src_ids': src_enc['input_ids'].squeeze(0),
            'tgt_ids': tgt_enc['input_ids'].squeeze(0),
            'src_text': src_text,
            'tgt_text': tgt_text
        }

# ==========================================
# 2. RNN 模型组件 (Encoder, Attention, Decoder)
# ==========================================

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, 
                          bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded)

class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method # 'dot', 'general', 'concat'
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        hidden_expanded = hidden.expand(encoder_output.size(0), encoder_output.size(1), self.hidden_size)
        energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_output), 2)))
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs, src_mask=None):
        if self.method == 'dot':
            attn_energies = self.dot_score(hidden.unsqueeze(1), encoder_outputs)
        elif self.method == 'general':
            attn_energies = self.general_score(hidden.unsqueeze(1), encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden.unsqueeze(1), encoder_outputs)

        if src_mask is not None:
            attn_energies = attn_energies.masked_fill(src_mask == 0, -1e9)
        return F.softmax(attn_energies, dim=1).unsqueeze(1) 

class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=2, attn_method='dot', dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.attn_method = attn_method

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, 
                          bidirectional=False, batch_first=True, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(attn_method, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs, src_mask=None):
        embedded = self.embedding(input_step)
        
        # 1. Forward through GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        
        # 2. Calculate Attention Weights using the output of GRU
        attn_weights = self.attn(rnn_output.squeeze(1), encoder_outputs, src_mask) 
        
        # 3. Calculate Context Vector
        context = attn_weights.bmm(encoder_outputs) 
        
        # 4. Concatenate GRU output and Context
        rnn_output = rnn_output.squeeze(1)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # 5. Final Output
        output = self.out(concat_output) 
        output = F.log_softmax(output, dim=1)
        
        return output, hidden, attn_weights

class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: [batch, src_len], tgt: [batch, tgt_len]
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        
        # Mask for attention (where src is not pad)
        src_mask = (src != self.pad_idx) 
        
        # First input to decoder is the <pad> token (or start token) from tgt
        decoder_input = tgt[:, 0].unsqueeze(1)
        
        # Use encoder's final hidden state as decoder's initial hidden state
        decoder_hidden = hidden
        
        for t in range(1, tgt_len):
            output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs, src_mask)
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs

# ==========================================
# 3. 实验控制器
# ==========================================

class ExperimentRunnerRNN:
    def __init__(self, train_path, val_path, test_path, output_dir="/final_proj"):
        # Local model path setup
        LOCAL_MODEL_PATH = "/final_proj/mt5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
        
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.results = {}
        self.bleu_scores = {}
        
        print("Loading datasets...")
        self.train_ds = TranslationDataset(train_path, self.tokenizer)
        self.val_ds = TranslationDataset(val_path, self.tokenizer)
        self.test_ds = TranslationDataset(test_path, self.tokenizer)
        print("Datasets loaded.")

    # --- Greedy Search Decoding ---
    def greedy_decode(self, model, src, max_len=50):
        model.eval()
        with torch.no_grad():
            src = src.unsqueeze(0).to(device)
            encoder_outputs, hidden = model.module.encoder(src)
            src_mask = (src != model.module.pad_idx)
            
            # Start with Pad/Start Token
            decoder_input = torch.tensor([[self.tokenizer.pad_token_id]], device=device)
            decoder_hidden = hidden
            
            decoded_ids = []
            
            for _ in range(max_len):
                output, decoder_hidden, _ = model.module.decoder(decoder_input, decoder_hidden, encoder_outputs, src_mask)
                top1 = output.argmax(1)
                
                if top1.item() == self.tokenizer.eos_token_id:
                    break
                decoded_ids.append(top1.item())
                decoder_input = top1.unsqueeze(1)
                
            return self.tokenizer.decode(decoded_ids, skip_special_tokens=True)

    # --- Beam Search Decoding ---
    def beam_search_decode(self, model, src, beam_width=3, max_len=50):
        model.eval()
        with torch.no_grad():
            src = src.unsqueeze(0).to(device)
            encoder_outputs, hidden = model.module.encoder(src)
            src_mask = (src != model.module.pad_idx)
            
            # Node: (score, input_token, hidden_state, sequence_list)
            start_node = (0, torch.tensor([[self.tokenizer.pad_token_id]], device=device), hidden, [])
            nodes = [start_node]
            
            final_candidates = []
            
            for _ in range(max_len):
                next_nodes = []
                for score, decoder_input, decoder_hidden, seq in nodes:
                    output, next_hidden, _ = model.module.decoder(decoder_input, decoder_hidden, encoder_outputs, src_mask)
                    # output is log_softmax
                    log_probs, indices = torch.topk(output, beam_width)
                    
                    for i in range(beam_width):
                        idx = indices[0][i].item()
                        prob = log_probs[0][i].item()
                        
                        if idx == self.tokenizer.eos_token_id:
                            final_candidates.append((score + prob, seq))
                        else:
                            next_nodes.append((score + prob, 
                                               torch.tensor([[idx]], device=device), 
                                               next_hidden, 
                                               seq + [idx]))
                
                # Prune beam
                next_nodes.sort(key=lambda x: x[0], reverse=True)
                nodes = next_nodes[:beam_width]
                
                if not nodes: break
                
            # If no EOS found, take current best
            if not final_candidates:
                final_candidates = [(n[0], n[3]) for n in nodes]
                
            best_score, best_seq = sorted(final_candidates, key=lambda x: x[0], reverse=True)[0]
            return self.tokenizer.decode(best_seq, skip_special_tokens=True)

    def train_rnn(self, exp_name, config):
        print(f"\n>>> [Train RNN]: {exp_name} | Attn: {config['attn_method']} | TF Ratio: {config['tf_ratio']}")
        
        loader = DataLoader(self.train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(self.val_ds, batch_size=config['batch_size'])
        
        # Init Model
        enc = EncoderRNN(self.tokenizer.vocab_size, config['hidden_size'], num_layers=2).to(device)
        dec = DecoderRNN(self.tokenizer.vocab_size, config['hidden_size'], num_layers=2, attn_method=config['attn_method']).to(device)
        model = Seq2SeqRNN(enc, dec, device, self.tokenizer.pad_token_id).to(device)
        model = torch.nn.DataParallel(model) 

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.NLLLoss(ignore_index=self.tokenizer.pad_token_id)
        
        val_hist = []
        best_val_loss = float('inf')
        save_path = os.path.join(self.logs_dir, f"GRU_{exp_name}_best.pth")
        
        for epoch in range(config['epochs']):
            model.train()
            epoch_loss = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['epochs']}[Train]")
            for batch in pbar:
                src = batch['src_ids'].to(device)
                tgt = batch['tgt_ids'].to(device)
                
                optimizer.zero_grad()
                
                output = model(src, tgt, teacher_forcing_ratio=config['tf_ratio'])
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
            # Validation
            model.eval()
            val_loss = 0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
            with torch.no_grad():
                for batch in val_pbar:
                    src = batch['src_ids'].to(device)
                    tgt = batch['tgt_ids'].to(device)
                    output = model(src, tgt, teacher_forcing_ratio=0) 
                    output_dim = output.shape[-1]
                    output = output[:, 1:].reshape(-1, output_dim)
                    tgt = tgt[:, 1:].reshape(-1)
                    val_loss += criterion(output, tgt).item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_hist.append(avg_val_loss)
            print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(loader):.4f} | Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.module.state_dict(), save_path)
                
        self.results[exp_name] = val_hist
        
        # Load Best Model for Evaluation
        model.module.load_state_dict(torch.load(save_path))
        return model

    def evaluate_test(self, model, exp_name, strategies=['greedy']):
        print(f"Evaluating {exp_name} on Test Set...")
        
        for strategy in strategies:
            sys_preds = []
            refs = []
            results_to_save = []
            
            indices = range(len(self.test_ds)) 
            pbar = tqdm(indices, desc=f"Decoding ({strategy})")
            
            for idx in pbar:
                src_ids = self.test_ds[idx]['src_ids']
                src_text = self.test_ds[idx]['src_text']
                tgt_text = self.test_ds[idx]['tgt_text']
                
                if strategy == 'greedy':
                    pred_text = self.greedy_decode(model, src_ids)
                elif strategy == 'beam':
                    pred_text = self.beam_search_decode(model, src_ids, beam_width=3)
                
                sys_preds.append(pred_text)
                refs.append(tgt_text)
                results_to_save.append({"src": src_text, "ref": tgt_text, "pred": pred_text})
            
            # Compute BLEU
            bleu = sacrebleu.corpus_bleu(sys_preds, [refs])
            score_key = f"{strategy}"
            if exp_name not in self.bleu_scores: self.bleu_scores[exp_name] = {}
            self.bleu_scores[exp_name][score_key] = bleu.score
            
            # Save logs
            log_file = os.path.join(self.logs_dir, f"GRU_{exp_name}_{strategy}_test_translation.jsonl")
            with open(log_file, 'w', encoding='utf-8') as f:
                for line in results_to_save:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            print(f"Saved {strategy} results. BLEU: {bleu.score:.2f}")

            with open("final_proj/logs/rnn_report.jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(f"Experiment {exp_name}, Strategy: {strategy}, Bleu: {bleu.score}", ensure_ascii=False) + "\n")

    def plot_results(self):
        # Plot Loss Curves
        plt.figure(figsize=(18, 12))
        for name, hist in self.results.items():
            plt.plot(hist, label=name)
        plt.title("RNN Validation Loss Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("NLL Loss")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "rnn_loss_comparison.png"))
        print("Loss plot saved.")
        
        # Print BLEU Table
        print("\n" + "="*60)
        print(f"{'Experiment':<25} | {'Greedy BLEU':<12} | {'Beam BLEU':<12}")
        print("-" * 60)
        
        for name, scores in self.bleu_scores.items():
            g_score = scores.get('greedy', '-')
            b_score = scores.get('beam', '-')
            if isinstance(g_score, float): g_score = f"{g_score:.2f}"
            if isinstance(b_score, float): b_score = f"{b_score:.2f}"
            print(f"{name:<25} | {g_score:<12} | {b_score:<12}")
        print("="*60 + "\n")

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    runner = ExperimentRunnerRNN(
        "/train_100k.jsonl", 
        "/valid.jsonl", 
        "/test.jsonl"
    )
    
    base_config = {
        'hidden_size': 256,
        'lr': 0.0001,
        'batch_size': 32,
        'epochs': 10,
        'attn_method': 'dot',
        'tf_ratio': 1.0 # Teacher Forcing (Baseline)
    }

    # === 实验 1: Attention Mechanism Ablation ===
    # 1.1 Dot Product (Baseline)
    model_dot = runner.train_rnn("Attn_Dot", base_config)
    runner.evaluate_test(model_dot, "Attn_Dot", strategies=['greedy'])
    
    # 1.2 Multiplicative (General)
    cfg_gen = base_config.copy(); cfg_gen['attn_method'] = 'general'
    model_gen = runner.train_rnn("Attn_General", cfg_gen)
    runner.evaluate_test(model_gen, "Attn_General", strategies=['greedy'])
    
    # 1.3 Additive (Concat)
    cfg_con = base_config.copy(); cfg_con['attn_method'] = 'concat'
    model_con = runner.train_rnn("Attn_Concat", cfg_con)
    runner.evaluate_test(model_con, "Attn_Concat", strategies=['greedy'])

    # === 实验 2: Training Policy (Teacher Forcing vs Free Running) ===
    cfg_free = base_config.copy()
    cfg_free['tf_ratio'] = 0.5 
    model_free = runner.train_rnn("Policy_Mix_TF0.5", cfg_free)
    runner.evaluate_test(model_free, "Policy_Mix_TF0.5", strategies=['greedy'])

    # === 实验 3: Decoding Policy (Greedy vs Beam) ===
    print("\n>>> Running Beam Search comparison on Baseline Model...")
    runner.evaluate_test(model_dot, "Attn_Dot_beam", strategies=['beam'])
    runner.evaluate_test(model_gen, "Attn_General_beam", strategies=['beam'])
    runner.evaluate_test(model_con, "Attn_Concat_beam", strategies=['beam'])

    # === 汇总报告 ===
    runner.plot_results()