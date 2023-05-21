import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The classes in the model are inspired from the model classes present in the blog : https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html suggested by you
In the blog everything has been done in context of translating sentences from the one language to other without using batches.
However I have understood the mechanism and applied it over for transliteration objective in the form of batches.
Some part of the code might look similar as I have applied a similar encoder, decoder and attention decoder architechture for my purpose.
'''

use_cuda = torch.cuda.is_available()

class EncoderRNN(nn.Module):
    
    # input_size -- size of input vocabulary dictionary
    # configuration -- consists of all hyperparameters
    def __init__(self, input_size, configuration):
        super(EncoderRNN, self).__init__()

        # embedding_size -- the size of each embedding vector
        self.embedding_size = configuration['embedding_size']
        # num_layers_encoder -- number of layers in encoder
        self.num_layers_encoder = configuration["num_layers_encoder"]
        # hidden_size -- The number of features in the hidden state
        self.hidden_size = configuration['hidden_size']
        # cell_type - one out of RNN, LSTM, GRU
        self.cell_type = configuration["cell_type"]
        # bidirectional -- input goes in normal and reverse order or just one order
        self.bi_directional = configuration['bi_directional']
        # drop_out -- the proabability p with which it randomly zeroes some of the elements of the input tensor
        self.drop_out = configuration['drop_out']
        
        # Initializing dropout layer with probability p
        self.dropout = nn.Dropout(self.drop_out)
        # Initializing embedding layer with input_size and embedding_size
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        
        # Initializing different cells
        # Bidirectional simply means to just putting two independent RNNs together. The input sequence is fed in normal time order for one network, and in reverse time order for another.
        self.cell_layer = None
        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = self.num_layers_encoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        elif self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = self.num_layers_encoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        else:
            self.cell_layer = nn.GRU(self.embedding_size, self.hidden_size, num_layers = self.num_layers_encoder, dropout = self.drop_out, bidirectional = self.bi_directional)
         
    def forward(self, input, batch_size, hidden):
        '''
            input -- as recieved a tensor
        '''
        # input is passed to embedding layer which converts them into embedding vectors and then reshapes them into a size of(1, batch_size, embedding_size)
        embedded = self.embedding(input).view(1,batch_size, -1)
        # the embedded output is then passed to dropout layer
        output = self.dropout(embedded)
        '''
            output -- the output out of dropout at every time step
            hidden -- the hidden states achieved at every time step
        '''
        # the output out of dropout is then passed into cell layer
        output, hidden = self.cell_layer(output, hidden)
        # returns the output and hidden state recieved from cell layer
        return output, hidden

    def initHidden(self ,batch_size, num_layers_enc):
        # Initializes the first hidden state to tensor of zeros
        res = None
        if self.bi_directional:
            # For bidirectional we double the hidden size
            res = torch.zeros(num_layers_enc* 2, batch_size, self.hidden_size)
        else:
            res = torch.zeros(num_layers_enc, batch_size, self.hidden_size)
        # Shift it to cuda if available 
        return res.cuda() if use_cuda else res

class DecoderRNN(nn.Module):
    # output_size -- size of target vocabulary dictionary
    # configuration -- consists of all hyperparameters
    def __init__(self, configuration,  output_size):
        super(DecoderRNN, self).__init__()

        # embedding_size -- the size of each embedding vector
        self.embedding_size = configuration['embedding_size']
        # hidden_size -- The number of features in the hidden state
        self.hidden_size = configuration['hidden_size']
        # num_layers_encoder -- number of layers in encoder        
        self.num_layers_decoder = configuration["num_layers_decoder"]
        # cell_type - one out of RNN, LSTM, GRU
        self.cell_type = configuration["cell_type"]
        # drop_out -- the proabability p with which it randomly zeroes some of the elements of the input tensor
        self.drop_out = configuration["drop_out"]
        # bidirectional -- input goes in normal and reverse order or just one order
        self.bi_directional = configuration["bi_directional"]

        # Initializing dropout layer with probability p
        self.dropout = nn.Dropout(self.drop_out)
        # Initializing embedding layer with input_size and embedding_size        
        self.embedding = nn.Embedding(output_size, self.embedding_size)

        # Initializing different cells
        # Bidirectional simply means to just putting two independent RNNs together. The input sequence is fed in normal time order for one network, and in reverse time order for another.
        self.cell_layer = None
        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        elif self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        else:
            self.cell_layer =   nn.GRU(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        
        # Initializing output layer
        if self.bi_directional:
            # For bidirectional we double the hidden size
            self.out = nn.Linear(self.hidden_size * 2 ,output_size)
        else:
            self.out = nn.Linear(self.hidden_size, output_size)
        
        # Apply softmax on output from output layer
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        '''
            input -- as recieved a tensor
        '''
        # input is passed to embedding layer which converts them into embedding vectors and then reshapes them into a size of(1, batch_size, embedding_size)
        output = self.embedding(input).view(1,batch_size, -1)
        # the embedded output is then passed to dropout layer        
        output = self.dropout(output)
        # the output out of dropout is then passed through relu activation
        output = F.relu(output)
        '''
            output -- the output out of dropout at every time step
            hidden -- the hidden states achieved at every time step
        '''
        # the output out of non linearity is then passed through cell layer
        output, hidden = self.cell_layer(output, hidden)
        # the output out of cell layer is passed to output layer
        output = self.out(output[0])
        # softmax is now applied on output out of output layer
        output = self.softmax(output)
        # returns the output and hidden state recieved from cell layer
        return output, hidden

class DecoderAttention(nn.Module) :
    # output_size -- size of target vocabulary dictionary
    # configuration -- consists of all hyperparameters
    def __init__(self, configuration, output_size) :

        super(DecoderAttention, self).__init__()
        
        # hidden_size -- The number of features in the hidden state
        self.hidden_size = configuration['hidden_size']
        # embedding_size -- the size of each embedding vector
        self.embedding_size = configuration['embedding_size']
        # cell_type - one out of RNN, LSTM, GRU
        self.cell_type = configuration['cell_type']
        # num_layers_encoder -- number of layers in encoder        
        self.num_layers_decoder = configuration['num_layers_decoder']
        # drop_out -- the proabability p with which it randomly zeroes some of the elements of the input tensor
        self.drop_out = configuration['drop_out']
        # max_length_word -- maximum length of words out of all input, target words
        self.max_length_word = configuration['max_length_word']
        # bidirectional -- input goes in normal and reverse order or just one order
        self.bi_directional = configuration["bi_directional"]

        # Initializing embedding layer with input_size and embedding_size  
        self.embedding = nn.Embedding(output_size, embedding_dim = self.embedding_size)
        # Initializing attention layer with embedding_size + hidden_size, max_length_word + 1
        self.attention_layer = nn.Linear(self.embedding_size + self.hidden_size, self.max_length_word + 1)
        if self.bi_directional:
            # For bidirectional we double the hidden size
            self.attention_combine = nn.Linear(self.embedding_size + self.hidden_size*2, self.embedding_size)
        else:
            self.attention_combine = nn.Linear(self.embedding_size + self.hidden_size, self.embedding_size)
        # Initializing dropout layer with probability p
        self.dropout = nn.Dropout(self.drop_out)

        # Initializing different cells
        # Bidirectional simply means to just putting two independent RNNs together. The input sequence is fed in normal time order for one network, and in reverse time order for another.
        self.cell_layer = None
        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        elif self.cell_type == 'GRU':
            self.cell_layer =   nn.GRU(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        else:
            self.cell_layer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = self.num_layers_decoder, dropout = self.drop_out, bidirectional = self.bi_directional)
        
        # Initializing output layer
        if self.bi_directional:
            # For bidirectional we double the hidden size
            self.out = nn.Linear(self.hidden_size * 2 ,output_size)
        else:
            self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, batch_size, hidden, encoder_outputs) :
        '''
            input -- as recieved a tensor
        '''
        # input is passed to embedding layer which converts them into embedding vectors and then reshapes them into a size of(1, batch_size, embedding_size)
        embedded = self.embedding(input).view(1, batch_size, -1)
        # Using the embedded output and hidden state we calculate attention weights by first passing them through attention layer and then applying softmax
        # Output obtained is then reshaped to (batch_size,1,self.max_length_word+1)
        attention_weights = None
        # For LSTM it is different as we have hidden as (hidden state, cell_state)
        if self.cell_type == 'LSTM' :
            attention_weights = F.softmax(self.attention_layer(torch.cat((embedded[0], hidden[0][0]), 1)), dim = 1).view(batch_size,1,self.max_length_word+1)
        
        else :
            attention_weights = F.softmax(self.attention_layer(torch.cat((embedded[0], hidden[0]), 1)), dim = 1).view(batch_size,1,self.max_length_word+1)

        # We then apply the attention to encoder outputs
        attention_applied = torch.bmm(attention_weights, encoder_outputs)
        # It is then reshaped to (1,batch_size,-1)
        attention_applied = attention_applied.view(1,batch_size,-1)
        # attention_applied output is then concatenated with embedded output
        output = torch.cat((embedded[0], attention_applied[0]), 1)
        # This output is passed through attention_combine layer
        output = self.attention_combine(output).unsqueeze(0)
        # The output from above is passed through relu non linearity
        output = F.relu(output)
        '''
            output -- the output out of dropout at every time step
            hidden -- the hidden states achieved at every time step
        '''
        # The output from above is then passed through cell layer
        output, hidden = self.cell_layer(output, hidden)
        # the output out of cell layer is passed to output layer
        output = self.out(output[0])
        # softmax is now applied on output out of output layer
        output = F.log_softmax(output, dim = 1)

        # returns the output and hidden state recieved from cell layer
        return output, hidden, attention_weights
