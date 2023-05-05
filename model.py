import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class EncoderRNN(nn.Module):

    def __init__(self, input_size, configuration):
        super(EncoderRNN, self).__init__()

        self.embedding_size = configuration['embedding_size']
        self.hidden_size = configuration['hidden_size']
        self.num_layers_encoder = configuration["num_layers_encoder"]
        self.cell_type = configuration["cell_type"]
        self.drop_out = configuration['drop_out']
        self.bi_directional = configuration['bi_directional']
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.dropout = nn.Dropout(self.drop_out)

        self.cell_layer = None
        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = self.num_layers_encoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        elif self.cell_type == 'GRU':
            self.cell_layer = nn.GRU(self.embedding_size, self.hidden_size, num_layers = self.num_layers_encoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        elif self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = self.num_layers_encoder, dropout = self.drop_out, bidirectional = self.bi_directional)
 
    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).view(1,batch_size, -1)
        embedded = self.dropout(embedded)
        output = embedded
        output, hidden = self.cell_layer(output, hidden)
        return output, hidden

    def initHidden(self ,batch_size, num_layers_enc):
        res = None
        if self.bi_directional:
            res = torch.zeros(num_layers_enc* 2, batch_size, self.hidden_size)
        else:
            res = torch.zeros(num_layers_enc, batch_size, self.hidden_size)
        if use_cuda : 
            return res.cuda()
        else :
            return res

class DecoderRNN(nn.Module):
    def __init__(self, configuration,  output_size):
        super(DecoderRNN, self).__init__()

        self.embedding_size = configuration['embedding_size']
        self.hidden_size = configuration['hidden_size']
        self.num_layers_decoder = configuration["num_layers_decoder"]
        self.cell_type = configuration["cell_type"]
        self.drop_out = configuration["drop_out"]
        self.bi_directional = configuration["bi_directional"]
        self.dropout = nn.Dropout(self.drop_out)
        self.embedding = nn.Embedding(output_size, self.embedding_size)

        self.cell_layer = None
        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        elif self.cell_type == 'GRU':
            self.cell_layer =   nn.GRU(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        elif self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        
        if self.bi_directional:
            self.out = nn.Linear(self.hidden_size * 2 ,output_size)
        else:
            self.out = nn.Linear(self.hidden_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        
        output = self.embedding(input).view(1,batch_size, -1)
        output = self.dropout(output)
        output = F.relu(output)
        output, hidden = self.cell_layer(output, hidden)
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden

class DecoderAttention(nn.Module) :

    def __init__(self, configs, output_size) :

        super(DecoderAttention, self).__init__()
        
        self.hidden_size = configs['hidden_size']
        self.embedding_size = configs['embedding_size']
        self.cell_type = configs['cell_type']
        self.num_layers_decoder = configs['num_layers_decoder']
        self.drop_out = configs['drop_out']
        self.max_length_word = configs['max_length_word']

        self.embedding = nn.Embedding(output_size, embedding_dim = self.embedding_size)
        self.attention_layer = nn.Linear(self.embedding_size + self.hidden_size, self.max_length_word + 1)
        self.attention_combine = nn.Linear(self.embedding_size + self.hidden_size, self.embedding_size)
        self.dropout = nn.Dropout(self.drop_out)

        self.cell_layer = None
        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out)
        elif self.cell_type == 'GRU':
            self.cell_layer =   nn.GRU(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out)
        elif self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, batch_size, hidden, encoder_outputs) :
        
        embedded = self.embedding(input).view(1, batch_size, -1)
        
        attention_weights = None
        if self.cell_type == 'LSTM' :
            attention_weights = F.softmax(self.attention_layer(torch.cat((embedded[0], hidden[0][0]), 1)), dim = 1)
        
        else :
            attention_weights = F.softmax(self.attention_layer(torch.cat((embedded[0], hidden[0]), 1)), dim = 1)

        attention_applied = torch.bmm(attention_weights.view(batch_size,1,self.max_length_word+1), encoder_outputs).view(1,batch_size,-1)
        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.cell_layer(output, hidden)
        output = self.out(output[0])
        output = F.log_softmax(output, dim = 1)
        
        return output, hidden, attention_weights
