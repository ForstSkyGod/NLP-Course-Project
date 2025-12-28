import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, MT5Tokenizer, MT5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset
import math
import json
import os
os.environ["WANDB_DISABLED"] = "true"
import matplotlib.pyplot as plt
import numpy as np
import random
import sacrebleu 
from tqdm import tqdm 

# ==========================================
# 0. 全局设置
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
# 1. 核心模块 (From Scratch Components)
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = d_model ** -0.5
        self.g = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        norm = torch.mean(x**2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.g

class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=32, max_distance=128, n_heads=8):
        super().__init__()
        self.num_buckets = num_buckets; self.max_distance = max_distance; self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)
    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
        else: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        val_if_large = max_exact + (torch.log(relative_position.float()/max_exact)/math.log(max_distance/max_exact)*(num_buckets-max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets-1))
        ret += torch.where(is_small, relative_position, val_if_large)
        return ret
    def forward(self, q_len, k_len):
        q_pos = torch.arange(q_len, dtype=torch.long, device=self.relative_attention_bias.weight.device)
        k_pos = torch.arange(k_len, dtype=torch.long, device=self.relative_attention_bias.weight.device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, bidirectional=True, num_buckets=self.num_buckets, max_distance=self.max_distance)
        return self.relative_attention_bias(rp_bucket).permute(2, 0, 1).unsqueeze(0)

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_head = d_model // n_heads; self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model); self.w_k = nn.Linear(d_model, d_model); self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model); self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value, mask=None, position_bias=None):
        batch_size = query.size(0)
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if position_bias is not None: scores += position_bias
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self.fc_out(torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_head))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, norm_type):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        NormClass = RMSNorm if norm_type == 'rmsnorm' else nn.LayerNorm
        self.norm1 = NormClass(d_model); self.norm2 = NormClass(d_model); self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None, pos_bias=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask, pos_bias))
        x = x + self.ffn(self.norm2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, norm_type):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = CustomMultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        NormClass = RMSNorm if norm_type == 'rmsnorm' else nn.LayerNorm
        self.norm1, self.norm2, self.norm3 = NormClass(d_model), NormClass(d_model), NormClass(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask, tgt_mask, self_pos_bias=None, cross_pos_bias=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask, self_pos_bias))
        x = x + self.dropout(self.cross_attn(self.norm2(x), enc_out, enc_out, src_mask, cross_pos_bias))
        x = x + self.ffn(self.norm3(x))
        return x

class NMTTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, num_layers=3, norm_type='layernorm', pos_type='absolute', dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.pos_type = pos_type
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = None; self.rel_pos_bias = None
        if pos_type == 'absolute':
            pe = torch.zeros(5000, d_model); position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
            pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
        elif pos_type == 'relative': 
            self.rel_pos_bias = RelativePositionBias(n_heads=n_heads)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_model*4, dropout, norm_type) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_model*4, dropout, norm_type) for _ in range(num_layers)])
        NormClass = RMSNorm if norm_type == 'rmsnorm' else nn.LayerNorm
        self.final_norm = NormClass(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size); 
        self.dropout = nn.Dropout(dropout)

    def create_masks(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones((tgt.size(1), tgt.size(1)), device=src.device)).bool()
        return src_mask, tgt_pad_mask & causal_mask

    def forward(self, src, tgt):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        if self.pos_type == 'absolute':
            src_emb = src_emb + self.pe[:, :src.size(1)]; tgt_emb = tgt_emb + self.pe[:, :tgt.size(1)]
        src_mask, tgt_mask = self.create_masks(src, tgt)
        enc_bias = self.rel_pos_bias(src.size(1), src.size(1)) if self.pos_type == 'relative' else None
        dec_bias = self.rel_pos_bias(tgt.size(1), tgt.size(1)) if self.pos_type == 'relative' else None
        
        enc_out = self.dropout(src_emb)
        for layer in self.encoder_layers: enc_out = layer(enc_out, src_mask, enc_bias)
        enc_out = self.final_norm(enc_out)
        
        dec_out = self.dropout(tgt_emb)
        for layer in self.decoder_layers: dec_out = layer(dec_out, enc_out, src_mask, tgt_mask, dec_bias, None)
        return self.fc_out(self.final_norm(dec_out))

    # --- 推理专用函数 ---
    def encode(self, src):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        if self.pos_type == 'absolute': src_emb = src_emb + self.pe[:, :src.size(1)]
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        enc_bias = self.rel_pos_bias(src.size(1), src.size(1)) if self.pos_type == 'relative' else None
        enc_out = self.dropout(src_emb)
        for layer in self.encoder_layers: enc_out = layer(enc_out, src_mask, enc_bias)
        return self.final_norm(enc_out), src_mask

    def decode_step(self, tgt, enc_out, src_mask):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        if self.pos_type == 'absolute': tgt_emb = tgt_emb + self.pe[:, :tgt.size(1)]
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones((tgt.size(1), tgt.size(1)), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & causal_mask
        dec_bias = self.rel_pos_bias(tgt.size(1), tgt.size(1)) if self.pos_type == 'relative' else None
        
        dec_out = self.dropout(tgt_emb)
        for layer in self.decoder_layers: dec_out = layer(dec_out, enc_out, src_mask, tgt_mask, dec_bias, None)
        return self.fc_out(self.final_norm(dec_out))

# ==========================================
# 2. 实验控制器
# ==========================================

class TranslationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=64):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 1. 读取数据并自动检测键名
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        
                        zn_key = next((k for k in obj.keys() if any(x in k.lower() for x in ['zn', 'zh', 'cn', 'source', 'src'])), None)
                        en_key = next((k for k in obj.keys() if any(x in k.lower() for x in ['en', 'eng', 'target', 'tgt'])), None)
                        
                        if zn_key and en_key:
                            # 只有当中英文都有值时才加入
                            if obj[zn_key] and obj[en_key]:
                                self.data.append({
                                    "zh": str(obj[zn_key]).strip(), 
                                    "en": str(obj[en_key]).strip()
                                })
                        
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"[Error] File not found: {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item['zh']
        tgt_text = item['en']
        
        if not src_text: src_text = " " 
        if not tgt_text: tgt_text = " "
        
        src_enc = self.tokenizer(src_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        tgt_enc = self.tokenizer(text_target=tgt_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        
        return {
            'src': src_enc['input_ids'].squeeze(0),
            'tgt': tgt_enc['input_ids'].squeeze(0),
            'src_text': src_text, 
            'tgt_text': tgt_text
        }

class ExperimentRunner:
    def __init__(self, train_path, val_path, test_path, output_dir="/root/data/250010050/code/HW/NLP/final_proj"):
        
        LOCAL_MODEL_PATH = "final_proj/mt5-small"
        self.tokenizer = MT5Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
        self.train_path, self.val_path, self.test_path = train_path, val_path, test_path
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

    def greedy_decode(self, model, src, max_len=50):
        model.eval()
        with torch.no_grad():
            enc_out, src_mask = model.module.encode(src)
            tgt = torch.tensor([[self.tokenizer.pad_token_id]], device=device)
            for _ in range(max_len):
                logits = model.module.decode_step(tgt, enc_out, src_mask)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id: break
            return tgt

    def beam_search_decode(self, model, src, beam_width=5, max_len=50):
        model.eval()
        with torch.no_grad():
            enc_out, src_mask = model.module.encode(src)
            candidates = [([self.tokenizer.pad_token_id], 0.0)]
            finished = []
            
            for _ in range(max_len):
                new_candidates = []
                for seq, score in candidates:
                    if seq[-1] == self.tokenizer.eos_token_id:
                        finished.append((seq, score))
                        continue
                        
                    tgt_tensor = torch.tensor([seq], device=device)
                    logits = model.module.decode_step(tgt_tensor, enc_out, src_mask)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                    topk_probs, topk_ids = torch.topk(log_probs, beam_width)
                    
                    for i in range(beam_width):
                        new_seq = seq + [topk_ids[i].item()]
                        new_score = score + topk_probs[i].item()
                        new_candidates.append((new_seq, new_score))
                
                ordered = sorted(new_candidates, key=lambda x: x[1], reverse=True)
                candidates = ordered[:beam_width]
                if not candidates: break 
            
            finished.extend(candidates)
            best_seq, best_score = sorted(finished, key=lambda x: x[1], reverse=True)[0]
            return torch.tensor([best_seq], device=device)

    def evaluate_test_set_and_save(self, model, exp_name, method='greedy', beam_width=5):
        print(f"Running Test Evaluation for {exp_name} ({method})...")
        results = []
        refs_for_bleu = []
        sys_for_bleu = []
        
        test_loader = tqdm(range(len(self.test_ds)), desc=f"Testing {exp_name}")
        
        for idx in test_loader:
            item = self.test_ds[idx]
            src = item['src'].unsqueeze(0).to(device)
            src_text = item['src_text']
            target_text = item['tgt_text']
            
            if method == 'greedy':
                out_ids = self.greedy_decode(model, src)
            else:
                out_ids = self.beam_search_decode(model, src, beam_width=beam_width)
            
            pred_text = self.tokenizer.decode(out_ids.squeeze(), skip_special_tokens=True)
            
            results.append({
                "src": src_text,
                "ref": target_text,
                "pred": pred_text
            })
            refs_for_bleu.append(target_text)
            sys_for_bleu.append(pred_text)
        
        # 1. 保存到 jsonl
        save_path = os.path.join(self.logs_dir, f"trans_{exp_name}_test_translation.jsonl")
        with open(save_path, 'w', encoding='utf-8') as f:
            for line in results:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"Test results saved to {save_path}")
        
        # 2. 计算 BLEU
        bleu = sacrebleu.corpus_bleu(sys_for_bleu, [refs_for_bleu])
        return bleu.score

    def train_from_scratch(self, exp_name, config):
        print(f"\n>>> [Train Scratch]: {exp_name}")
        loader = DataLoader(self.train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=config['batch_size'])
        
        model = NMTTransformer(self.tokenizer.vocab_size, config['d_model'], config['n_heads'], config['layers'], config['norm_type'], config['pos_type']).to(device)
        model = torch.nn.DataParallel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        val_hist = []
        best_val_loss = float('inf')
        save_path = os.path.join("/root/data/250010050/code/HW/NLP/final_proj", f"trans_{exp_name}_best.pth")
        
        for epoch in range(config['epochs']):
            model.train()
            total_loss = 0
            
            train_pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
            for batch in train_pbar:
                src, tgt = batch['src'].to(device), batch['tgt'].to(device)
                optimizer.zero_grad()
                out = model(src, tgt[:, :-1])
                loss = criterion(out.reshape(-1, self.tokenizer.vocab_size), tgt[:, 1:].reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            model.eval()
            val_loss_accum = 0
            val_steps = 0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
            with torch.no_grad():
                for batch in val_pbar:
                    src = batch['src'].to(device)
                    tgt = batch['tgt'].to(device)
                    out = model(src, tgt[:, :-1])
                    loss = criterion(out.reshape(-1, self.tokenizer.vocab_size), tgt[:, 1:].reshape(-1))
                    val_loss_accum += loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss_accum / val_steps if val_steps > 0 else 0
            val_hist.append(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"  >> Epoch {epoch+1}: New Best Val Loss ({best_val_loss:.4f}). Saving model to {save_path}...")
                
                real_model_to_save = model.module if hasattr(model, "module") else model
                torch.save(real_model_to_save.state_dict(), save_path)
            
            print(f"Epoch {epoch+1} Summary | Train Loss: {total_loss/len(loader):.4f} | Val Loss: {avg_val_loss:.4f}")

        self.results[exp_name] = val_hist
        
        bleu_score = self.evaluate_test_set_and_save(model, exp_name, method='greedy')
        self.bleu_scores[exp_name] = {'greedy': bleu_score}
        print(f"Experiment {exp_name} Final BLEU: {bleu_score:.2f}")
        
        with open("final_proj/logs/report.jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(f"Experiment {exp_name}; Method: greedy; Final BLEU: {bleu_score:.2f}", ensure_ascii=False) + "\n")

    def run_real_finetuning(self):
        print(f"\n>>> [Experiment]: Real Finetuning (mT5)")
        
        def load_jsonl_to_hf(path):
            data_list = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            # 自动匹配键名逻辑同 TranslationDataset
                            zn_key = next((k for k in obj.keys() if any(x in k.lower() for x in ['zn', 'zh', 'cn', 'source', 'src'])), None)
                            en_key = next((k for k in obj.keys() if any(x in k.lower() for x in ['en', 'eng', 'target', 'tgt'])), None)
                            if zn_key and en_key:
                                data_list.append({"src": str(obj[zn_key]), "tgt": str(obj[en_key])})
            return HFDataset.from_list(data_list)

        train_hf = load_jsonl_to_hf(self.train_path)
        val_hf = load_jsonl_to_hf(self.val_path)
        test_hf = load_jsonl_to_hf(self.test_path) # 加载测试集用于最终评估

       
        def preprocess(examples):
            inputs = examples['src']
            targets = examples['tgt']
            model_inputs = self.tokenizer(inputs, max_length=64, truncation=True)
            labels = self.tokenizer(text_target=targets, max_length=64, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_train = train_hf.map(preprocess, batched=True)
        tokenized_val = val_hf.map(preprocess, batched=True)
        tokenized_test = test_hf.map(preprocess, batched=True)

        
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple): preds = preds[0]
        
            preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)

            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]
            
            result = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
            return {"bleu": result.score}

        LOCAL_MODEL_PATH = "final_proj/mt5-small"
        model = MT5ForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH).to(device)
        
        args = Seq2SeqTrainingArguments(
            output_dir = "/final_proj",
            eval_strategy="epoch",      # 每个 epoch 评估一次
            save_strategy="epoch",            # 每个 epoch 保存一次
            learning_rate=2e-4,
            fp16=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,               # 保持和其他实验一致
            predict_with_generate=True,       # 【关键】开启生成模式，否则无法计算 BLEU
            generation_max_length=200,         # 生成最大长度
            logging_steps=10,
            load_best_model_at_end=True,      # 训练结束加载最好模型
            metric_for_best_model="bleu",     # 以 BLEU 为标准选择最好模型
            save_total_limit=1                # 只保留一个最好的 checkpoint
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=model),
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics   
        )
        
        trainer.train()
        trainer.save_model(LOCAL_MODEL_PATH)
        
        
        eval_losses = [x['eval_loss'] for x in trainer.state.log_history if 'eval_loss' in x]
        if not eval_losses: eval_losses = [0] 
        self.results["Finetuned_mT5"] = eval_losses
        
        print("Evaluating Finetuned model on Test Set...")
        test_metrics = trainer.evaluate(tokenized_test, metric_key_prefix="test")
        final_bleu = test_metrics['test_bleu']
        
        self.bleu_scores["Finetuned_mT5"] = {'beam': final_bleu}
        
        print(f"Finetuned mT5 Test BLEU: {final_bleu:.2f}")

        predict_results = trainer.predict(tokenized_test, metric_key_prefix="test")
        if isinstance(predict_results.predictions, tuple):
            preds = predict_results.predictions[0]
        else:
            preds = predict_results.predictions
        
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        src_texts = [ex['src'] for ex in test_hf]
        ref_texts = [ex['tgt'] for ex in test_hf]
        
        with open("/root/data/250010050/code/HW/NLP/final_proj/logs/report.jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(f"Finetuned mT5 Test BLEU: {final_bleu:.2f}", ensure_ascii=False) + "\n")

        save_path = os.path.join(self.logs_dir, "Finetuned_mT5_test_translation.jsonl")
        with open(save_path, 'w', encoding='utf-8') as f:
            for src, ref, pred in zip(src_texts, ref_texts, decoded_preds):
                f.write(json.dumps({"src": src, "ref": ref, "pred": pred.strip()}, ensure_ascii=False) + "\n")
        print(f"Test predictions saved to {save_path}")

    def plot_and_report(self, excludeNames=None): 
        # Plot Loss
        plt.figure(figsize=(18, 12))
        for name, hist in self.results.items():
            if name in excludeNames:
                continue
            else:
                plt.plot(hist, label=name)
        plt.title("Validation Loss Comparison")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "loss_comparison.png"))
        
        # Print BLEU Report
        print("\n" + "="*50)
        print(f"{'Experiment':<25} | {'BLEU':<10}")
        print("-" * 50)
        for name, scores in self.bleu_scores.items():
            score = list(scores.values())[0]
            print(f"{name:<25} | {score:<10.2f}")
        print("="*50 + "\n")

    def plot_group(self, exp_names, title, filename):
        plt.figure(figsize=(18, 12))
        has_data = False
        for name in exp_names:
            if name in self.results:
                plt.plot(self.results[name], label=name, marker='o', markersize=3)
                has_data = True
            else:
                print(f"Warning: Result for {name} not found, skipping plot.")
        
        if has_data:
            plt.title(title)
            plt.xlabel("Epochs")
            plt.ylabel("Validation Loss")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            plt.close()

# ==========================================
# 3. 主入口
# ==========================================
if __name__ == "__main__":
    runner = ExperimentRunner(
        "/train_100k.jsonl", 
        "/valid.jsonl", 
        "/test.jsonl"
    )
    
    # 基础配置
    base_cfg = {'batch_size': 16, 'lr': 1e-3, 'epochs': 100, 'd_model': 64, 'n_heads': 8, 'layers': 2, 'norm_type': 'layernorm', 'pos_type': 'absolute'}
    
    # 1. Baseline
    runner.train_from_scratch("Baseline", base_cfg)
    
    # 2. Ablation: RMSNorm
    cfg_rms = base_cfg.copy()
    cfg_rms['norm_type'] = 'rmsnorm'
    runner.train_from_scratch("RMSNorm", cfg_rms)
    
    # 3. Ablation: Relative Pos
    cfg_rel = base_cfg.copy()
    cfg_rel['pos_type'] = 'relative'
    runner.train_from_scratch("Scratch_RelPos", cfg_rel)

    # 4. Hyperparameter Sensitivity 
    bs_list = [32, 64, 128]
    bs_exp_names = []
    
    for bs in bs_list:
        exp_name = f"Sens_BS_{bs}"
        bs_exp_names.append(exp_name)
        
        cfg = base_cfg.copy()
        cfg['batch_size'] = bs
        runner.train_from_scratch(exp_name, cfg)
    
    runner.plot_group(bs_exp_names, "Sensitivity: Batch Size", "sensitivity_batch_size.png")

    
    lr_list = [1e-4, 5e-5, 1e-5]
    lr_exp_names = []
    
    for lr in lr_list:
        exp_name = f"Sens_LR_{lr}"
        lr_exp_names.append(exp_name)
        
        cfg = base_cfg.copy()
        cfg['lr'] = lr
        runner.train_from_scratch(exp_name, cfg)
    
    runner.plot_group(lr_exp_names, "Sensitivity: Learning Rate", "sensitivity_learning_rate.png")


    scale_list = [128, 256, 512]
    scale_exp_names = []
    
    for d_model in scale_list:
        exp_name = f"Sens_Scale_{d_model}"
        scale_exp_names.append(exp_name)
        
        cfg = base_cfg.copy()
        cfg['d_model'] = d_model
        
            
        runner.train_from_scratch(exp_name, cfg)
        
    runner.plot_group(scale_exp_names, "Sensitivity: Model Scale (d_model)", "sensitivity_model_scale.png")

    # 5.T5微调
    runner.run_real_finetuning()

    skipnames = []
    skipnames.extend(bs_exp_names)
    skipnames.extend(lr_exp_names)
    skipnames.extend(scale_exp_names)

    runner.plot_and_report(excludeNames=skipnames)