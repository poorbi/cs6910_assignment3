import os
import wandb
import torch
import random
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

dir = 'aksharantar_sampled'
use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
UNK_token = 3
PAD_token = 4

class Vocabulary:

    def __init__(self, name):
        self.char2count = {}
        self.char2index = {}
        self.n_chars = 4
        self.index2char = {0: '<', 1: '>',2 : '?', 3:'.'}
        self.name = name

    def addWord(self, word):
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.index2char[self.n_chars] = char
                self.char2count[char] = 1
                self.n_chars += 1
            else:
                self.char2count[char] += 1
        

def prepareData(dir, lang1, lang2):

    data = pd.read_csv(dir,sep=",",names=['input', 'target'])
    max_input_length = max([len(txt) for txt in data['input'].to_list()])
    max_target_length = max([len(txt) for txt in data['target'].to_list()])

    input_lang = Vocabulary(lang1)
    output_lang = Vocabulary(lang2)

    pairs = []
    input_list,target_list = data['input'].to_list(),data['target'].to_list()
    for i in range(len(input_list)):
        pairs.append([input_list[i],target_list[i]])

    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])

    prepared_data = {
        'input_lang' : input_lang,
        'output_lang' : output_lang,
        'pairs' : pairs,
        'max_input_length' : max_input_length,
        'max_target_length' : max_target_length
    }

    return prepared_data

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
        embedded = self.dropout(self.embedding(input).view(1,batch_size, -1))
        output = embedded
        output, hidden = self.cell_layer(output, hidden)
        return output, hidden

    def initHidden(self ,batch_size):
        res = None
        if self.bi_directional:
            res = torch.zeros(self.num_layers_encoder * 2, batch_size, self.hidden_size)
        else:
            res = torch.zeros(self.num_layers_encoder, batch_size, self.hidden_size)
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
        
        output = self.dropout(self.embedding(input).view(1,batch_size, -1))
        output = F.relu(output)
        output, hidden = self.cell_layer(output, hidden)
        
        output = self.softmax(self.out(output[0]))
        return output, hidden

    # def initHidden(self, batch_size):
    #     res = torch.zeros(self.num_layers_decoder, batch_size, self.hidden_size)
    #     if use_cuda : 
    #         return res.cuda()
    #     else :
    #         return res

def indexesFromWord(lang, word):
    index_list = []
    for char in word:
        if char in lang.char2index.keys():
            index_list.append(lang.char2index[char])
        else:
            index_list.append(UNK_token)
    return index_list

def variableFromSentence(lang, word, max_length):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    indexes.extend([PAD_token] * (max_length - len(indexes)))
    result = torch.LongTensor(indexes)
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPairs(input_lang, output_lang, pairs, max_length):
    res = []
    for pair in pairs:
        input_variable = variableFromSentence(input_lang, pair[0], max_length)
        target_variable = variableFromSentence(output_lang, pair[1], max_length)
        res.append((input_variable, target_variable))
    return res

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length, teacher_forcing_ratio = 0.5):
    
    batch_size = configuration['batch_size']
    encoder_hidden = encoder.initHidden(batch_size)

    input_tensor = Variable(input_tensor.transpose(0, 1))
    target_tensor = Variable(target_tensor.transpose(0, 1))

    if configuration["cell_type"] == "LSTM":
        encoder_cell_state = encoder.initHidden(batch_size)
        encoder_hidden = (encoder_hidden, encoder_cell_state)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], batch_size, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([SOS_token]*batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden= decoder(decoder_input, batch_size, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, batch_size,decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = torch.cat(tuple(topi))

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_tensor[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
  
def cal_val_loss(encoder, decoder, input_tensor, target_tensor, configuration, criterion , max_length):

    with torch.no_grad():

        batch_size = configuration['batch_size']

        encoder_hidden = encoder.initHidden(batch_size)

        input_tensor = Variable(input_tensor.transpose(0, 1))
        target_tensor = Variable(target_tensor.transpose(0, 1))
            
        if configuration["cell_type"] == "LSTM":
            encoder_cell_state = encoder.initHidden(batch_size)
            encoder_hidden = (encoder_hidden, encoder_cell_state)

        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0
            
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], batch_size, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, batch_size, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = torch.cat(tuple(topi))

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output, target_tensor[di])

    return loss.item() / target_length

def evaluate(encoder, decoder, loader, configuration, criterion , max_length):

    with torch.no_grad():

        batch_size = configuration['batch_size']
        total = 0
        correct = 0
        
        for batch_x, batch_y in loader:

            encoder_hidden = encoder.initHidden(batch_size)

            input_variable = Variable(batch_x.transpose(0, 1))
            target_variable = Variable(batch_y.transpose(0, 1))
            
            if configuration["cell_type"] == "LSTM":
                encoder_cell_state = encoder.initHidden(batch_size)
                encoder_hidden = (encoder_hidden, encoder_cell_state)

            input_length = input_variable.size()[0]
            target_length = target_variable.size()[0]

            output = torch.LongTensor(target_length, batch_size)

            encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
            
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_variable[ei], batch_size, encoder_hidden)

            decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_hidden = encoder_hidden

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, batch_size, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                decoder_input = torch.cat(tuple(topi))
                output[di] = torch.cat(tuple(topi))

            output = output.transpose(0,1)
            for di in range(output.size()[0]):
                ignore = [SOS_token, EOS_token, PAD_token]
                sent = [configuration['output_lang'].index2char[letter.item()] for letter in output[di] if letter not in ignore]
                y = [configuration['output_lang'].index2char[letter.item()] for letter in batch_y[di] if letter not in ignore]
                if sent == y:
                    correct += 1
                total += 1

    return (correct/total)*100

def trainIters(encoder, decoder, train_loader, val_loader, test_loader, learning_rate, configuration, wandb_flag):

    max_length = configuration['max_length_word']

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [],[],[],[]

    encoder_optimizer, decoder_optimizer = None, None

    if configuration['optimizer']=='nadam':
        encoder_optimizer = optim.NAdam(encoder.parameters(),lr=learning_rate)
        decoder_optimizer = optim.NAdam(decoder.parameters(),lr=learning_rate)
    else:
        encoder_optimizer = optim.Adam(encoder.parameters(),lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(),lr=learning_rate)

    criterion = nn.NLLLoss()
    
    ep = 15

    for i in range(ep):
        
        if i % 5 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time = ", current_time)

        train_loss_total = 0
        val_loss_total = 0

        for batchx, batchy in train_loader:
            loss = None

            if configuration['attention'] == False:
                loss = train(batchx, batchy, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length)
            
            train_loss_total += loss
        
        train_loss_total = train_loss_total/len(train_loader)
        print('ep : ', i, ' | ', end='')
        print('train loss :', train_loss_total, ' | ', end='')

        for batchx, batchy in val_loader:
            loss = None

            if configuration['attention'] == False:
                loss = cal_val_loss(encoder, decoder, batchx, batchy, configuration, criterion , max_length)
            
            val_loss_total += loss

        val_loss_total = val_loss_total/len(val_loader)
        train_acc = evaluate(encoder, decoder, train_loader, configuration, criterion, max_length)
        val_acc = evaluate(encoder, decoder, val_loader, configuration, criterion, max_length)

        train_loss_list.append(train_loss_total)
        val_loss_list.append(val_loss_total)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        
        print("train accuracy : " ,train_acc, ' | ', end='')
        print('val loss :', val_loss_total, ' | ', end='')
        print("val accuracy : " ,val_acc)

        if wandb_flag == True:
            wandb.log({
                "train_loss"           : train_loss_total,
                "validation_loss"      : val_loss_total,
                "train_accuracy"       : train_acc,
                "validation_accuracy"  : val_acc
                })

    temp = configuration['batch_size']
    configuration['batch_size'] = 1
    print("test accuracy for the model : " ,evaluate(encoder, decoder, test_loader, configuration, criterion, max_length))
    configuration['batch_size'] = temp

def main():

    configuration = {
            "hidden_size" : 256,
            "source_lang" : 'eng',
            "target_lang" : 'hin',
            "cell_type"   : 'LSTM',
            "num_layers_encoder" : 2,
            "num_layers_decoder" : 2,
            "drop_out"    : 0.2, 
            "embedding_size" : 256,
            "bi_directional" : True,
            "batch_size" : 32,
            "attention" : False ,
            "optimizer" : 'nadam',
        }
    
    train_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_train.csv')
    validation_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_valid.csv')
    test_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_test.csv')

    train_prepared_data= prepareData(train_path,configuration['source_lang'], configuration['target_lang'])

    input_lang = train_prepared_data['input_lang']
    output_lang = train_prepared_data['output_lang']
    pairs = train_prepared_data['pairs']
    max_input_length = train_prepared_data['max_input_length']
    max_target_length = train_prepared_data['max_target_length']
    
    val_prepared_data= prepareData(validation_path,configuration['source_lang'], configuration['target_lang'])

    val_pairs = val_prepared_data['pairs']
    max_input_length_val = val_prepared_data['max_input_length']
    max_target_length_val = val_prepared_data['max_target_length']

    test_prepared_data= prepareData(validation_path, configuration['source_lang'], configuration['target_lang'])

    test_pairs = test_prepared_data['pairs']
    max_input_length_test = test_prepared_data['max_input_length']
    max_target_length_test = test_prepared_data['max_target_length']

    max_list = [max_input_length, max_target_length, max_input_length_val, max_target_length_val, max_input_length_test, max_target_length_test]
    max_len_all = max(max_list)

    max_len = max(max_input_length, max_target_length) + 2

    configuration['input_lang'] = input_lang
    configuration['output_lang'] = output_lang
    configuration['max_length_word'] = max_len_all + 1

    learning_rate = 0.001

    encoder1 = EncoderRNN(input_lang.n_chars, configuration)
    decoder1 = DecoderRNN(configuration, output_lang.n_chars)
    if use_cuda:
        encoder1=encoder1.cuda()
        decoder1=decoder1.cuda()

    pairs = variablesFromPairs(configuration['input_lang'], configuration['output_lang'], pairs , configuration['max_length_word'])
    val_pairs = variablesFromPairs(configuration['input_lang'], configuration['output_lang'], val_pairs, configuration['max_length_word'])
    test_pairs = variablesFromPairs(configuration['input_lang'], configuration['output_lang'], test_pairs, configuration['max_length_word'])

    train_loader = torch.utils.data.DataLoader(pairs, batch_size=configuration['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_pairs, batch_size=configuration['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_pairs, batch_size=1, shuffle=True)

    if configuration['attention'] == False :
        trainIters(encoder1, decoder1, train_loader, val_loader, test_loader, learning_rate, configuration,False)

main()
