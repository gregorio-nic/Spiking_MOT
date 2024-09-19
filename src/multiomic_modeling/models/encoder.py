import torch
from torch import nn
from multiomic_modeling.models.utils.embedding import Embeddings, PositionalEncoding
from multiomic_modeling.models.utils import init_params_xavier_uniform, init_params_xavier_normal, EncoderState
from multiomic_modeling.data.structs import Sequence
from multiomic_modeling import logging
import snntorch as snn
from snntorch import functional as SF
from multiomic_modeling.models.snn_transformer import TransformerEncoderLayerSNN, TransformerEncoderSNN

logger = logging.create_logger(__name__)


class TorchSeqTransformerEncoder(nn.Module):
    def __init__(self, d_input, d_model=1024, d_ff=1024, n_heads=16, n_layers=2, dropout=0.1, beta_enc = 0.3, thr_enc = 2.0):
        super(TorchSeqTransformerEncoder, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.n_layers = n_layers
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Linear(self.d_input, self.d_model)
        self.beta_enc = beta_enc 
        self.thr_enc = thr_enc
        self.encoder_layer = TransformerEncoderLayerSNN(self.d_model, self.n_heads, self.d_ff, self.dropout, activation="relu")
        #encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout, activation="relu")
        encoder_norm = nn.LayerNorm(d_model)
        self.net = TransformerEncoderSNN(self.encoder_layer, self.n_layers, norm=encoder_norm)
        #self.net = nn.TransformerEncoder(encoder_layer, self.n_layers, encoder_norm)
        self.s_enc = snn.Leaky(threshold=self.thr_enc, beta=self.beta_enc, learn_beta=True, learn_threshold=True)
        init_params_xavier_uniform(self)

    def forward(self, inputs) -> EncoderState:
        mem_enc = self.s_enc.init_leaky()
        mask_padding_x = ~inputs[1]
        inputs = inputs[0].float()
        self.net.init_hidden_leaky()
        
        x = self.embedding(inputs)
        x = x.transpose(0, 1)
        for _ in range(10):
            memory = self.net(x, src_key_padding_mask=mask_padding_x)
            #print(memory)
            spk_enc, mem_enc = self.s_enc(memory, mem_enc)
        return EncoderState(memory=spk_enc, mask_padding_x=mask_padding_x)
