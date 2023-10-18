import os
import math
import torch
import numpy as np
from data_load import *
import torch.nn as nn

vocab_size = 100
embedding_dimension = 512
ffn_hiddenlayer_dimension = 2048
dk = dv = 64
coder_num = 6
head_num = 8


src_vocab, idx2cn = load_cn_vocab()
tgt_vocab, idx2en = load_en_vocab()
tgt_vocab_size = len(tgt_vocab)
src_vocab_size = len(src_vocab)
enc_inputs, dec_outputs, dec_inputs = load_test_data()
enc_inputs = torch.LongTensor(enc_inputs)
dec_inputs = torch.LongTensor(dec_inputs)
dec_outputs = torch.LongTensor(dec_outputs)


# 位置编码
class PositionalEncode(nn.Module):
    def __init__(self, class_embedding_dimension=embedding_dimension, dropout=0.1, max_len=50):
        super(PositionalEncode, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encode = torch.zeros(max_len, class_embedding_dimension)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        encode = torch.exp(torch.arange(0, class_embedding_dimension, 2).float() * (
                -math.log(10000.0) / class_embedding_dimension))
        positional_encode[:, 0::2] = torch.sin(position * encode)
        positional_encode[:, 1::2] = torch.cos(position * encode)

        positional_encode = positional_encode.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_encode', positional_encode)

    def forward(self, x):
        """
        x: [seq_len, batch_size, embedding_dimension]
        """

        output = x + self.positional_encode[:x.size(0), :]

        return self.dropout(output)


# 填充句掩码
def PadMask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    """
    batch_size, len_q, = seq_q.size()
    batch_size, len_k, = seq_k.size()

    pad_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_mask.expand(batch_size, len_q, len_k)


# 多头注意力掩码
def AttentionMask(seq):
    mask_shape = [seq.size(0), seq.size(1), seq.size(1)]

    attention_mask = np.triu(np.ones(mask_shape), k=1)
    attention_mask = torch.from_numpy(attention_mask).byte()

    return attention_mask


# 获取注意力权重
class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        attention_weight = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(dk)
        attention_weight.masked_fill_(mask, -1e9)
        attention_weight = self.softmax(attention_weight)
        context = torch.matmul(attention_weight, V)
        return context, attention_weight


# 多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(embedding_dimension, dk * head_num, bias=False)
        self.WK = nn.Linear(embedding_dimension, dk * head_num, bias=False)
        self.WV = nn.Linear(embedding_dimension, dv * head_num, bias=False)
        self.fc = nn.Linear(head_num * dv, embedding_dimension, bias=False)
        self.layernorm = nn.LayerNorm(embedding_dimension).cuda()

    def forward(self, input_Q, input_K, input_V, mask):
        """
        input_Q: [batch_size, len_q, embedding_dimension]
        input_K: [batch_size, len_k, embedding_dimension]
        input_V: [batch_size, len_v(=len_k), embedding_dimension]
        attn_mask: [batch_size, seq_len, seq_len]
        """

        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.WQ(input_Q).view(batch_size, -1, head_num, dk).transpose(1, 2)
        K = self.WK(input_K).view(batch_size, -1, head_num, dk).transpose(1, 2)
        V = self.WV(input_V).view(batch_size, -1, head_num, dv).transpose(1, 2)

        mask = mask.unsqueeze(1).repeat(1, head_num, 1, 1)
        context, attention = DotProductAttention()(Q, K, V, mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, head_num * dv)

        output = self.fc(context)

        return self.layernorm(output + residual), attention


# 全连接层
class FeedForwardLayer(nn.Module):
    def __init__(self):
        super(FeedForwardLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dimension, ffn_hiddenlayer_dimension, bias=False),
            nn.ReLU(),
            nn.Linear(ffn_hiddenlayer_dimension, embedding_dimension, bias=False)
        )
        self.layernorm = nn.LayerNorm(embedding_dimension).cuda()

    def forward(self, x):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = x
        output = self.fc(x)

        return self.layernorm(output + residual)


# 构造编码层
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.attentionLayer = MultiHeadAttention()
        self.feedforwardLayer = FeedForwardLayer()

    def forward(self, x, mask):
        """
        x: [batch_size, src_len, d_model]
        mask: [batch_size, src_len, src_len]
        """
        x, attention = self.attentionLayer(x, x, x, mask)
        output = self.feedforwardLayer(x)

        return output, attention


# 组装编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embeddingLayer = nn.Embedding(src_vocab_size, embedding_dimension)
        self.position = PositionalEncode(class_embedding_dimension=embedding_dimension)
        self.encodeLayer = nn.ModuleList([EncoderLayer() for _ in range(coder_num)])

    def forward(self, x):
        """
        x: [batch_size, src_len]
        """

        x1 = self.embeddingLayer(x)
        x1 = self.position(x1.transpose(0, 1)).transpose(0, 1)
        mask = PadMask(x, x)
        attentions = []
        for layer in self.encodeLayer:
            output, attention = layer(x1, mask)
            attentions.append(attention)

        return output, attentions


# 构造解码层
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.attentionLayer = MultiHeadAttention()
        self.other_attentionLayer = MultiHeadAttention()
        self.feedforwardLayer = FeedForwardLayer()

    def forward(self, encoder_x, x, mask, attention_mask):
        """
        x: [batch_size, src_len, d_model]
        encoder_x: [batch_size, src_len, d_model]
        mask: [batch_size, src_len, src_len]
        attention_mask: [batch_size, tgt_len, tgt_len]
        """
        x, self_attention = self.attentionLayer(x, x, x, attention_mask)
        x, attention = self.other_attentionLayer(x, encoder_x, encoder_x, mask)
        output = self.feedforwardLayer(x)

        return output, attention, self_attention


# 组装解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embeddingLayer = nn.Embedding(tgt_vocab_size, embedding_dimension)
        self.position = PositionalEncode(embedding_dimension)
        self.decodeLayer = nn.ModuleList([DecoderLayer() for _ in range(coder_num)])

    def forward(self, x, encoder_input, encoder_x):
        """
        x: [batch_size, src_len]
        encoder_x: [batch_size, src_len, d_model]
        encoder_input: [batch_size, src_len]
        """

        x1 = self.embeddingLayer(x)
        x1 = self.position(x1.transpose(0, 1)).transpose(0, 1).cuda()
        mask = PadMask(x, x).cuda()
        attention_mask = AttentionMask(x).cuda()
        attention_mask = torch.gt((mask + attention_mask), 0).cuda()
        ed_mask = PadMask(x, encoder_input)

        self_attentions, attentions = [], []
        for layer in self.decodeLayer:
            output, attention, self_attention = layer(x1, encoder_x, attention_mask, ed_mask)
            attentions.append(attention)
            self_attentions.append(self_attention)

        return output, attentions, self_attentions


# 构造Transformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.linear = nn.Linear(embedding_dimension, tgt_vocab_size, bias=False).cuda()

    def forward(self, encoder_x, decoder_x):
        encoder_output, encoder_attentions = self.encoder(encoder_x)
        decoder_output, decoder_attentions, decoder_self_attentions = self.decoder(decoder_x, encoder_x, encoder_output)
        output = self.linear(decoder_output)

        return output.view(-1, output.size(-1)), encoder_attentions, decoder_attentions, decoder_self_attentions

"""
初始化模型加载
"""
try:
    print('===loading model===')
    current_project_path = os.getcwd()
    transformer = Transformer().cuda()
    checkpoint_path = os.path.join(current_project_path,
                                   'weight/weight200_loss0.09.pth')
    if torch.cuda.is_available():
        transformer.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0'))
    else:
        transformer.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    transformer.eval()
    print('===model lode sucessed===')

except Exception as e:
    print('===model load error:{}==='.format(e))


enc_inputs, dec_inputs = enc_inputs.cuda(), dec_inputs.cuda()
output, encoder_attentions, decoder_attentions, decoder_self_attentions = transformer(enc_inputs, dec_inputs)
# output = torch.softmax(output, dim=-1)
output = torch.argmax(output, dim=-1)

sentence = []
for idx in output:
    word = idx2en[idx.item()]
    sentence.append(word)

print(sentence)




