import torch
from torch import nn
from multiomic_modeling.models.utils.embedding import Embeddings, PositionalEncoding
from multiomic_modeling.models.utils import init_params_xavier_uniform, init_params_xavier_normal,  EncoderState, generate_padding_mask
from multiomic_modeling import logging
from multiomic_modeling.data.structs import Sequence
import snntorch as snn
from snntorch import functional as SF
from multiomic_modeling.models.snn_transformer import TransformerDecoderLayerSNN, TransformerDecoderSNN

logger = logging.create_logger(__name__)


class TorchSeqTransformerDecoderOriginal(nn.Module):
    def __init__(self, nb_classes, d_model=1024, d_ff=1024, n_heads=16, n_layers=2, dropout=0.1, activation="relu",beta_dec = 0.2, thr_dec = 2.0 ): #dff = 4 * dmodel
        super(TorchSeqTransformerDecoderOriginal, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.nb_classes = nb_classes
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.activation = activation
        
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.n_layers, norm=decoder_norm)

        self.output = nn.Linear(d_model, self.nb_classes)

        init_params_xavier_uniform(self)
        # init_params_xavier_normal(self)
        
    def forward(self, enc_state: EncoderState):
        target = torch.zeros((1, enc_state.memory.shape[1], self.d_model), device=enc_state.memory.device)

        x = self.decoder(
            target,
            enc_state.memory,
            memory_mask=enc_state.mask_x,
            memory_key_padding_mask=enc_state.mask_padding_x,
        )
        x = self.output(x)[0]
        return x
    
    

class TorchSeqTransformerDecoderViewsOriginal(nn.Module):
    def __init__(self, d_input, d_model=1024, d_ff=1024, n_heads=16, n_layers=2, dropout=0.1, activation="relu", thr_out = 0.2, beta_out = 6): #dff = 4 * dmodel
        super(TorchSeqTransformerDecoderViewsOriginal, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_input = d_input
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.activation = activation
        self.thr_out = thr_out
        self.beta_out = beta_out
        
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.n_layers, norm=decoder_norm)

        self.output = nn.Linear(d_model, self.d_input)

        init_params_xavier_uniform(self)
        
    def forward(self, enc_state: EncoderState):
        target = torch.zeros((3, enc_state.memory.shape[1], self.d_model), device=enc_state.memory.device) # change this to 5 or 3 
        
        x = self.decoder(
            target,
            enc_state.memory,
            memory_mask=enc_state.mask_x,
            memory_key_padding_mask=enc_state.mask_padding_x,
        )
        x = self.output(x)
        
        
        return x