import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import math
import os
import json

# ==========================================
# 0. 全局设置与工具
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_weights_safe(model, path):
    """安全加载权重，自动处理 DataParallel 的 module. 前缀"""
    if not os.path.exists(path):
        print(f"[Warning] Model file not found: {path}")
        return False
    
    print(f"Loading weights from {path}...")
    state_dict = torch.load(path, map_location=device)
    
    # 如果是直接保存的 state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 'module.' 前缀 (如果是用 DataParallel 训练的)
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    try:
        model.load_state_dict(new_state_dict)
        model.eval()
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"[Error] Failed to load weights: {e}")
        return False

# ==========================================
# 1. RNN 模型定义
# ==========================================
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, 
                          bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method 
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
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, 
                          bidirectional=False, batch_first=True, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(attn_method, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs, src_mask=None):
        embedded = self.embedding(input_step)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output.squeeze(1), encoder_outputs, src_mask) 
        context = attn_weights.bmm(encoder_outputs)
        rnn_output = rnn_output.squeeze(1)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
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

# ==========================================
# 2. Transformer 模型定义 
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
# 3. 推理引擎
# ==========================================
class InferenceEngine:
    def __init__(self, tokenizer_path):
        print(f"Loading Tokenizer from {tokenizer_path}...")
        try:
            self.tokenizer = MT5Tokenizer.from_pretrained(tokenizer_path)
        except:
            print(f"Local path failed, trying 'google/mt5-small'...")
            self.tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
        self.pad_idx = self.tokenizer.pad_token_id
        self.eos_idx = self.tokenizer.eos_token_id

    def load_rnn(self, model_path, hidden_size=256, attn_method='dot'):
        enc = EncoderRNN(self.tokenizer.vocab_size, hidden_size, num_layers=2).to(device)
        dec = DecoderRNN(self.tokenizer.vocab_size, hidden_size, num_layers=2, attn_method=attn_method).to(device)
        model = Seq2SeqRNN(enc, dec, device, self.pad_idx).to(device)
        
        if load_weights_safe(model, model_path):
            self.model_type = 'rnn'
            self.model = model
            return True
        return False

    def load_transformer_scratch(self, model_path, d_model=64, n_heads=8, layers=2, norm_type='layernorm', pos_type='absolute'):
        model = NMTTransformer(self.tokenizer.vocab_size, d_model, n_heads, layers, norm_type, pos_type).to(device)
        
        if load_weights_safe(model, model_path):
            self.model_type = 'transformer_scratch'
            self.model = model
            return True
        return False

    def load_hf_mt5(self, model_path):
        """加载微调后的 mT5"""
        if not os.path.exists(model_path):
            print("HF Model path invalid.")
            return False
        if os.path.isdir(model_path):
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
            self.model_type = 'hf_mt5'
            return True
        elif model_path.endswith('.pth'):
            base_model_path = "final_proj/mt5-small" 
            self.model = MT5ForConditionalGeneration.from_pretrained(base_model_path).to(device)
            if load_weights_safe(self.model, model_path):
                self.model_type = 'hf_mt5'
                return True
        return False

    def translate(self, text, strategy='greedy', beam_width=3, max_len=64):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True).input_ids.to(device)
        
        # 1. HF MT5 Inference
        if self.model_type == 'hf_mt5':
            outputs = self.model.generate(
                inputs, 
                max_length=max_len, 
                num_beams=(beam_width if strategy=='beam' else 1),
                early_stopping=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 2. RNN Inference (Greedy Only impl for brevity, extendable)
        elif self.model_type == 'rnn':
            # RNN Greedy Logic
            with torch.no_grad():
                encoder_outputs, hidden = self.model.encoder(inputs)
                src_mask = (inputs != self.pad_idx)
                
                decoder_input = torch.tensor([[self.pad_idx]], device=device)
                decoder_hidden = hidden
                decoded_ids = []
                
                for _ in range(max_len):
                    output, decoder_hidden, _ = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs, src_mask)
                    top1 = output.argmax(1)
                    if top1.item() == self.eos_idx: break
                    decoded_ids.append(top1.item())
                    decoder_input = top1.unsqueeze(1)
                
                return self.tokenizer.decode(decoded_ids, skip_special_tokens=True)

        # 3. Scratch Transformer Inference
        elif self.model_type == 'transformer_scratch':
            with torch.no_grad():
                enc_out, src_mask = self.model.encode(inputs)
                tgt = torch.tensor([[self.pad_idx]], device=device)
                
                # Greedy
                if strategy == 'greedy':
                    for _ in range(max_len):
                        logits = self.model.decode_step(tgt, enc_out, src_mask)
                        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                        if next_token.item() == self.eos_idx: break
                        tgt = torch.cat([tgt, next_token], dim=1)
                    return self.tokenizer.decode(tgt.squeeze(), skip_special_tokens=True)
                
                # Simple Beam implementation could go here
                # Defaulting to greedy for brevity in unified file
                else:
                    print("Warn: Beam search for scratch transformer not fully implemented in this unified snippet, using greedy.")
                    for _ in range(max_len):
                        logits = self.model.decode_step(tgt, enc_out, src_mask)
                        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                        if next_token.item() == self.eos_idx: break
                        tgt = torch.cat([tgt, next_token], dim=1)
                    return self.tokenizer.decode(tgt.squeeze(), skip_special_tokens=True)

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # --- 配置区域 (修改这里) ---
    BASE_DIR = "/final_proj"
    TOKENIZER_PATH = os.path.join(BASE_DIR, "mt5-small")
    
    # 待测试的输入文本
    test_sentences = [
        "这是一个测试句子。",
        "我喜欢学习自然语言处理。",
        "今天天气真不错。"
    ]
    
    # 初始化引擎
    engine = InferenceEngine(TOKENIZER_PATH)

    print("\n" + "="*40)
    print(" >>> 测试模式选择 <<<")
    print("1. RNN (Attn_Dot)")
    print("2. Transformer (Baseline Scratch)")
    print("3. HF Finetuned mT5")
    choice = input("请输入数字选择模型 (1/2/3): ").strip()
    
    success = False
    
    if choice == '1': # (不推荐，因为只训了10轮很烂)
        # 加载 RNN 权重
        rnn_path = os.path.join(BASE_DIR, "GRU_Attn_Dot_best.pth")
        success = engine.load_rnn(rnn_path, hidden_size=256, attn_method='dot')
        
    elif choice == '2':
        # 加载 Transformer Scratch 权重
        trans_path = os.path.join(BASE_DIR, "trans_Sens_BS_64_best.pth")
        success = engine.load_transformer_scratch(trans_path, d_model=64, n_heads=8, layers=2)
        
    elif choice == '3':
        # 加载 mT5
        mt5_path = "final_proj/mt5-small"
        success = engine.load_hf_mt5(mt5_path)

    # --- 执行预测 ---
    if success:
        print(f"\nModel Loaded: {engine.model_type}")
        print("-" * 40)
        for text in test_sentences:
            translation = engine.translate(text, strategy='greedy')
            print(f"Src: {text}")
            print(f"Tgt: {translation}")
            print("-" * 20)
            
        # 交互模式
        while True:
            user_input = input("\n请输入中文 (输入 'q' 退出): ")
            if user_input.lower() == 'q': break
            translation = engine.translate(user_input)
            print(f">> En: {translation}")
    else:
        print("模型加载失败，请检查路径配置。")