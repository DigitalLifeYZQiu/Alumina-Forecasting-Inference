from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import numpy as np

from models.modules.decoder import Decoder, DecoderLayer
from models.modules.encoder import Encoder, EncoderLayer

from models.modules.attn import FullAttention, AttentionLayer
from models.modules.embed import DataEmbedding, DateEmbedding, CycleTimeEmbedding, DataEmbedding_inverted,DataEmbedding_inverted_new
import pdb


class iTransformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, d_model=512, n_heads=8,
                 e_layers=3, d_layers=2, d_ff=512, dropout=0.5, activation='gelu', output_attention=False,
                 device=0, timestamp_feature=True, stage='train',**kwargs):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.device = device
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.c_out = c_out
        self.timestamp_feature = timestamp_feature
        self.stage = stage

        # Encoding
        # & original enc_embedding
        # self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, 'timeF', 'h', dropout)
        # self.enc_embedding = DataEmbedding_inverted_new(seq_len, d_model, 'timeF', 'h', dropout)
        self.enc_embedding = nn.ModuleList([DataEmbedding_inverted(seq_len, d_model, 'timeF', 'h', dropout) for i in range(self.enc_in)])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, self.pred_len, bias=True)

        # self.target_projector = nn.Linear(self.enc_in, self.c_out, bias=True)
        # self.target_projector = nn.Linear(self.enc_in, 1, bias=True)

        self.global_step = 0

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        if self.timestamp_feature != 'none':
            x_mark_enc = None

        # & original enc_embedding calculation
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # & channel independent enc_embedding calculation
        enc_out_single_array=[]
        for index in range(self.enc_in):
            x_enc_single = x_enc[:,:,index].unsqueeze(2)
            enc_out_single = self.enc_embedding[index](x_enc_single,x_mark_enc)
            enc_out_single_array.append(enc_out_single)
        enc_out = torch.cat(enc_out_single_array,dim=1)
        

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates 

        # if self.stage=='train':
        #     dec_out = self.target_projector(dec_out)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out[:,:,:self.c_out]

