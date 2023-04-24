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

MAX_LENGTH = 28
TAR_MAX_LENGTH = 28
SOS_token = 0
EOS_token = 1
lang2_ = 'hin'
drop = 0.2

class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2

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

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    pairs = []
    input_list,target_list = data['input'].to_list(),data['target'].to_list()
    for i in range(len(input_list)):
        pairs.append([input_list[i],target_list[i]])

    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    
    print("Counted words:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs

dir = 'aksharantar_sampled'
train_path = os.path.join(dir, lang2_, lang2_ + '_train.csv')
validation_path = os.path.join(dir, lang2_, lang2_ + '_valid.csv')
test_path = os.path.join(dir, lang2_, lang2_ + '_test.csv')

input_lang, output_lang, pairs = prepareData(train_path,'eng', 'hin')
val_input_lang, val_output_lang, val_pairs = prepareData(validation_path,'eng', 'hin')
print(random.choice(pairs))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(drop)
        self.gru = nn.GRU(hidden_size, hidden_size,num_layers = 1)

    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input).view(1, 1, -1))
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = 1)
        self.dropout = nn.Dropout(drop)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input).view(1, 1, -1))
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def evaluate(encoder, decoder, word, target, criterion, max_length=TAR_MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromWord(input_lang, word)
        target_tensor = tensorFromWord(output_lang, target)
        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_chars = []
        loss = 0
        str_word = ''

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                str_word = str_word.join(decoded_chars)
                decoded_chars.append('<EOS>')
                break
            else:
                decoded_chars.append(output_lang.index2char[topi.item()])
            if(di < target_length) : 
                loss += criterion(decoder_output, target_tensor[di])
            decoder_input = topi.squeeze().detach()

        

        return loss.item()/target_length, str_word

def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in word]


def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromWord(input_lang, pair[0])
    target_tensor = tensorFromWord(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden= decoder(
                decoder_input, decoder_hidden)
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

def trainIters(encoder, decoder, n_iters, learning_rate=0.01):
    print(len(pairs))
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = []
    for i in range(n_iters) :
        training_pairs.append(tensorsFromPair(pairs[i]))
    criterion = nn.CrossEntropyLoss()
    plot_losses = []
    val_plot_losses = []
    
    ep = 5
    for i in range(ep):
        print(i)
        val_loss = 0
        plot_loss_total = 0
        for iter in range(1, n_iters + 1):
            pair_ = pairs[iter - 1]
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            plot_loss_total += loss
        
        count = 0 
        train_acc = 0
        char_train_acc = 0
        for pair_ in pairs:
            _ , out_str_ = evaluate(encoder,decoder,pair_[0],pair_[1],criterion)
            print(count , " : ", pair_[0]," ",pair_[1]," ", out_str_)
            if(out_str_ == pair_[1]):
                train_acc+=1
            for i in range(len(out_str_)):
                char_acc = 0
                if(out_str_[i] == pair_[1][i]):
                     char_acc+= 1
                char_acc /= (len(out_str_))
                char_train_acc += char_acc 
            count+=1
         
        val_acc = 0
        char_val_acc = 0
        count = 0 
        for val_pair in val_pairs:
            v_loss, out_str = evaluate(encoder,decoder,val_pair[0],val_pair[1],criterion)
            if(out_str == val_pair[1]):
                val_acc+=1
            for i in range(len(out_str)):
                char_acc = 0
                if(out_str[i] == val_pair[1][i]):
                     char_acc+= 1
                char_acc /= (len(out_str))
                char_val_acc += char_acc
            val_loss += v_loss 
            count+=1
    
        val_loss = val_loss/len(val_pairs)

        print('train loss :',plot_loss_total/n_iters)
        print("train accuracy : ", train_acc/len(pairs))
        print("char wise train accuracy : ", char_train_acc/len(pairs))
        print('validation loss : ',val_loss)
        print("char wise validation accuracy : ", char_val_acc/len(val_pairs))
        plot_losses.append(plot_loss_total/n_iters)
        val_plot_losses.append(val_loss)

    print('before plot')
    # showPlot(plot_losses)
    showPlot(val_plot_losses)
    print('after loss')

def showPlot(points):
    plt.figure()
    plt.plot(points)
    plt.show()
    
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_chars, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_chars).to(device)

trainIters(encoder1, decoder1,len(pairs))
