import numpy as np
import pandas as pd
import os
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SOS_token = 0
EOS_token = 1
UNK_token = 3
PAD_token = 4
lang2_ = 'hin'

class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: '<', 1: '>',2 : '?', 3:'.'}
        self.n_chars = 4

    def addWord(self, word):
        for char in word:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

def prepareData(dir, lang1, lang2):

    data = pd.read_csv(dir,sep=",",names=['input', 'target'])

    max_input_length = max([len(txt) for txt in data['input'].to_list()])

    max_target_length = max([len(txt) for txt in data['target'].to_list()])

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    pairs = []
    input_list,target_list = data['input'].to_list(),data['target'].to_list()
    for i in range(len(input_list)):
        pairs.append([input_list[i],target_list[i]])

    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    
    print("Counted letters:")
    print(input_lang.name, max_input_length)
    print(output_lang.name, max_target_length)
    return input_lang, output_lang, pairs, max_input_length, max_target_length

class EncoderRNN(nn.Module):
    def __init__(self, input_size, configuration):
        super(EncoderRNN, self).__init__()

        self.embedding_size = configuration['embedding_size']
        self.hidden_size = configuration['hidden_size']
        self.num_layers = configuration["num_layers"]
        self.cell_type = configuration["cell_type"]
        self.drop_out = configuration["drop_out"]
        self.bi_directional = configuration["bi_directional"]

        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.dropout = nn.Dropout(self.drop_out)
        self.cell_layer = None
        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = self.num_layers, dropout = self.drop_out)
        elif self.cell_type == 'GRU':
            self.cell_layer = nn.GRU(self.embedding_size, self.hidden_size, num_layers = self.num_layers, dropout = self.drop_out)
        elif self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = self.num_layers, dropout = self.drop_out)
 
    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input).view(1, 1, -1))
        output, hidden = self.cell_layer(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, configuration,  output_size):
        super(DecoderRNN, self).__init__()

        self.embedding_size = configuration['embedding_size']
        self.hidden_size = configuration['hidden_size']
        self.num_layers = configuration["num_layers"]
        self.cell_type = configuration["cell_type"]
        self.drop_out = configuration["drop_out"]
        self.bi_directional = configuration["bi_directional"]
        self.dropout = nn.Dropout(self.drop_out)

        self.embedding = nn.Embedding(output_size, self.embedding_size)

        self.cell_layer = None
        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = self.num_layers, dropout = self.drop_out)
        elif self.cell_type == 'GRU':
            self.cell_layer =   nn.GRU(self.embedding_size, self.hidden_size, num_layers = self.num_layers, dropout = self.drop_out)
        elif self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = self.num_layers, dropout = self.drop_out)
        
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input).view(1, 1, -1))
        output = F.relu(output)
        output, hidden = self.cell_layer(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

dir = '/kaggle/input/aksharantar-sampled/aksharantar_sampled'
train_path = os.path.join(dir, lang2_, lang2_ + '_train.csv')
validation_path = os.path.join(dir, lang2_, lang2_ + '_valid.csv')
test_path = os.path.join(dir, lang2_, lang2_ + '_test.csv')

input_lang, output_lang, pairs, max_input_length, max_target_length = prepareData(train_path,'eng', 'hin')
max_len = max(max_input_length, max_target_length) + 1
val_input_lang, val_output_lang, val_pairs, u, w = prepareData(validation_path,'eng', 'hin')
test_input_lang, test_output_lang, test_pairs, test_u, test_w = prepareData(test_path,'eng', 'hin')
print(random.choice(pairs))

def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in word]

def tensorFromWord(lang, type, word):
    indexes = indexesFromWord(lang, word)
    len_padding = 0

    if type == "input":
        len_padding = max_input_length - len(indexes) + 1
    if lang == "target":
        len_padding = max_target_length - len(indexes) + 1

    indexes.append(EOS_token)
    for i in range(len_padding):
        indexes.append(PAD_token)
    
    return torch.tensor(indexes, dtype = torch.long, device = device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromWord(input_lang,'input', pair[0])
    target_tensor = tensorFromWord(output_lang,'target', pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length= max_len):
    encoder_hidden = encoder.initHidden()

    if configuration["cell_type"] == "LSTM":
        encoder_cell_state = encoder.initHidden()
        encoder_hidden = (encoder_hidden, encoder_cell_state)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate(encoder, decoder, word, target, criterion, max_length = max_len):
    with torch.no_grad():
        input_tensor = tensorFromWord(input_lang,'input', word)
        target_tensor = tensorFromWord(output_lang,'target', target)
        
        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        if configuration["cell_type"] == "LSTM":
          encoder_cell_state = encoder.initHidden()
          encoder_hidden = (encoder_hidden, encoder_cell_state)

        # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            # encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_chars = []
        loss = 0
        str_word = ''

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                str_word = str_word.join(decoded_chars)
                break
            else:
                decoded_chars.append(output_lang.index2char[topi.item()])
            if(di < target_length) : 
                loss += criterion(decoder_output, target_tensor[di])
            decoder_input = topi.squeeze().detach()

        return loss.item()/target_length, str_word

def trainIters(encoder, decoder, n_iters, learning_rate, configuration):

    train_plot_losses = []
    val_plot_losses = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = []
    for i in range(n_iters) :
        training_pairs.append(tensorsFromPair(pairs[i]))

    criterion = nn.CrossEntropyLoss()
    
    ep = 10

    for i in range(ep):
        print(i)
        val_loss = 0
        plot_loss_total = 0
        print('training..')
        for iter in range(1, n_iters + 1):
            if(iter%5120 == 0):
              print(iter)
            op = pairs[iter - 1]
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration)
            plot_loss_total += loss
        
        print('calculating train acc ..')
        count = 0 
        train_acc = 0
        for pair_ in pairs:
            if(count%5120 == 0):
              print(count)
            _ , out_str_ = evaluate(encoder,decoder,pair_[0],pair_[1],criterion)
            if(out_str_ == pair_[1]):
                train_acc+=1
            count+=1

        print('calculating val acc ..') 
        val_acc = 0
        count = 0 
        for val_pair in val_pairs:
            if(count%512 == 0):
              print(count)
            v_loss, out_str = evaluate(encoder,decoder,val_pair[0],val_pair[1],criterion)
#             print(count, ':', val_pair[0],' ', val_pair[1],' ',out_str)
            if(out_str == val_pair[1]):
                val_acc+=1
            val_loss += v_loss 
            count+=1
    
        val_loss = val_loss/len(val_pairs)
        val_acc = val_acc/len(val_pairs)
        plot_loss_total = plot_loss_total/n_iters
        train_acc = train_acc/len(pairs)

        print('train loss :', plot_loss_total)
        print("train accuracy : ", train_acc)
        print('validation loss : ',val_loss)
        print('validation accuracy : ',val_acc)

        train_plot_losses.append(plot_loss_total/n_iters)
        val_plot_losses.append(val_loss)

    test_acc = 0
    count = 0 
    for test_pair in test_pairs:
        test_loss, test_out_str = evaluate(encoder,decoder,test_pair[0],test_pair[1],criterion)
        if(test_out_str == test_pair[1]):
            test_acc+=1 
        count+=1
    print('test acc :', test_acc)
    print('before plot')
    showPlot(val_plot_losses)
    print('after loss')

def showPlot(points):
    plt.figure()
    plt.plot(points)
    plt.show()

configuration = {
        "hidden_size" : 256,
        "input_lang" : 'eng',
        "target_lang" : 'hin',
        "cell_type"   : "GRU",
        "num_layers" : 2 ,
        "drop_out"    : 0.2, 
        "embedding_size" : 256,
        "bi_directional" : False,
        "batch_size" : 64,
        "attention" : False
    }

encoder1 = EncoderRNN(input_lang.n_chars, configuration).to(device)
decoder1 = DecoderRNN(configuration, output_lang.n_chars).to(device)

trainIters(encoder1, decoder1,len(pairs), 0.01, configuration)
