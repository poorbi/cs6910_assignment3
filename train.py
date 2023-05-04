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

parser=argparse.ArgumentParser()

parser.add_argument('-wp',      '--wandb_project',      help='project name',                                                    type=str,       default='CS6910_A3_')
parser.add_argument('-we',      '--wandb_entity',       help='entity name',                                                     type=str,       default='cs22m064'  )
parser.add_argument('-e',       '--epochs',             help='epochs',                          choices=[5,10],                 type=int,       default=10          )
parser.add_argument('-b',       '--batch_size',         help='batch sizes',                     choices=[32,64,128],            type=int,       default=64          )
parser.add_argument('-o',       '--optimizer',          help='optimizer',                       choices=['adam','nadam'],       type=str,       default='nadam'     )
parser.add_argument('-lr',      '--learning_rate',      help='learning rates',                  choices=[1e-2,1e-3],            type=float,     default=1e-3        )
parser.add_argument('-nle',     '--num_layers_en',      help='number of layers in encoder',     choices=[1,2,3],                type=int,       default=3           )
parser.add_argument('-nld',     '--num_layers_dec',     help='number of layers in decoder',     choices=[1,2,3],                type=int,       default=2           )
parser.add_argument('-sz',      '--hidden_size',        help='hidden layer size',               choices=[32,64,256,512],        type=int,       default=512         )
parser.add_argument('-il',      '--input_lang',         help='input language',                  choices=['eng'],                type=str,       default='eng'       )
parser.add_argument('-tl',      '--target_lang',        help='target language',                 choices=['hin','tel'],          type=str,       default='hin'       )
parser.add_argument('-ct',      '--cell_type',          help='cell type',                       choices=['LSTM','GRU','RNN'],   type=str,       default='LSTM'      )
parser.add_argument('-do',      '--drop_out',           help='drop out',                        choices=[0.0,0.2,0.3],          type=float,     default=0.2         )
parser.add_argument('-es',      '--embedding_size',     help='embedding size',                  choices=[16,32,64,256],         type=int,       default=128         )
parser.add_argument('-bd',      '--bidirectional',      help='bidirectional',                   choices=[True,False],           type=bool,      default=False       )
parser.add_argument('-at',      '--attention',          help='attention',                       choices=[True,False],           type=bool,      default=False       )

args=parser.parse_args()

project_name_ap     = args.wandb_project
entity_name_ap      = args.wandb_entity
epochs_ap           = args.epochs
batch_size_ap       = args.batch_size
optimizer_ap        = args.optimizer
learning_rate_ap    = args.learning_rate
num_layers_en_ap    = args.num_layers_en
num_layers_dec_ap   = args.num_layers_dec
hidden_size_ap      = args.hidden_size
input_lang_ap       = args.input_lang
target_lang_ap      = args.target_lang
cell_type_ap        = args.cell_type
drop_out_ap         = args.drop_out
embedding_size_ap   = args.embedding_size
bidirectional_ap    = args.bidirectional
attention_ap        = args.attention

dir = 'aksharantar_sampled'
use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
UNK_token = 3
PAD_token = 4

sweep_config ={
    'method':'bayes'
}

metric = {
    'name' : 'validation_accuracy',
    'goal' : 'maximize'
}
sweep_config['metric'] = metric

parameters_dict={
    'epochs':{
        'values' : [5,10]
    },
    'hidden_size':{
        'values' : [128,256,512]
    },
    'cell_type':{
        'values' : ['LSTM','GRU','RNN']
    },
    'learning_rate':{
        'values' : [1e-2,1e-3]
    },
    'num_layers_en':{
        'values' : [1,2,3]
    },
    'num_layers_dec':{
        'values' : [1,2,3]
    },
    'drop_out':{
        'values' : [0.0,0.2,0.3]
    },
    'embedding_size':{
        'values' : [64,128,256]
    },
    'batch_size':{
        'values' : [32,64,128]
    },
    'optimizer':{
        'values' : ['adam','nadam']
    },
    'bidirectional':{
        'values' : [True,False]
    }
}
sweep_config['parameters'] = parameters_dict

class prepareVocabulary:
    def __init__(self, name):
        self.char2count = {}
        self.char2index = {}
        self.n_chars = 4
        self.index2char = {0: '(', 1: ')',2 : '?', 3:'_'}
        self.name = name

    def addWord(self, word):
        for i in range(len(word)):
            if word[i] not in self.char2index:
                self.char2index[word[i]] = self.n_chars
                self.index2char[self.n_chars] = word[i]
                self.char2count[word[i]] = 1
                self.n_chars += 1
            else:
                self.char2count[word[i]] += 1        

def prepareData(lang1, lang2, dir):

    data = pd.read_csv(dir,sep=",",names=['input', 'target'])

    max_input_length = max([len(inp) for inp in data['input'].to_list()])
    input_lang = prepareVocabulary(lang1)

    max_target_length = max([len(tar) for tar in data['target'].to_list()])
    output_lang = prepareVocabulary(lang2)

    pairs = []
    input_list,target_list = data['input'].to_list(),data['target'].to_list()

    for i in range(len(input_list)):
        pairs.append([input_list[i],target_list[i]])

    for i in range(len(pairs)):
        input_lang.addWord(pairs[i][0])
        output_lang.addWord(pairs[i][1])

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

def tensorFromWord(lang, word, max_length):
    index_list = []
    for i in range(len(word)):
        if word[i] in lang.char2index.keys():
            index_list.append(lang.char2index[word[i]])
        else:
            index_list.append(UNK_token)

    indexes = index_list
    indexes.append(EOS_token)

    indexes.extend([PAD_token] * (max_length - len(indexes)))

    result = torch.LongTensor(indexes)
    if use_cuda:
        return result.cuda()
    else:
        return result

def tensorFromPairs(input_lang, output_lang, pairs, max_length):
    res = []
    for i in range(len(pairs)):
        input_variable = tensorFromWord(input_lang, pairs[i][0], max_length)
        target_variable = tensorFromWord(output_lang, pairs[i][1], max_length)
        res.append((input_variable, target_variable))
    return res

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length, teacher_forcing_ratio = 0.5):
    
    batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
    encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

    input_tensor = input_tensor.transpose(0, 1)
    target_tensor = target_tensor.transpose(0, 1)

    if configuration["cell_type"] == "LSTM":
        encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], batch_size, encoder_hidden)

    decoder_input = torch.LongTensor([SOS_token]*batch_size)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True 
    else:
        use_teacher_forcing = False

    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden= decoder(decoder_input, batch_size, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i]

    else:
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, batch_size,decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_tensor[i])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
  
def cal_val_loss(encoder, decoder, input_tensor, target_tensor, configuration, criterion , max_length):

    with torch.no_grad():

        batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
        encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

        input_tensor = input_tensor.transpose(0, 1)
        target_tensor = target_tensor.transpose(0, 1)
            
        if configuration["cell_type"] == "LSTM":
            encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]

        loss = 0
            
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], batch_size, encoder_hidden)

        decoder_input = torch.LongTensor([SOS_token] * batch_size)
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, batch_size, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output, target_tensor[i])

    return loss.item() / target_length

def evaluate(encoder, decoder, loader, configuration, criterion , max_length):

    with torch.no_grad():

        total = 0
        correct = 0
        
        for batch_x, batch_y in loader:
            batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
            encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

            input_variable = batch_x.transpose(0, 1)
            target_variable = batch_y.transpose(0, 1)
            
            if configuration["cell_type"] == "LSTM":
                encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

            input_length = input_variable.size()[0]
            target_length = target_variable.size()[0]

            output = Variable(torch.LongTensor(target_length, batch_size))

            encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size)
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
            
            for i in range(input_length):
                encoder_output, encoder_hidden = encoder(input_variable[i], batch_size, encoder_hidden)

            decoder_input = torch.LongTensor([SOS_token] * batch_size)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_hidden = encoder_hidden

            for i in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, batch_size, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi
                output[i] = torch.cat(tuple(topi))

            output = output.transpose(0,1)
            for i in range(output.size()[0]):
                ignore = [SOS_token, EOS_token, PAD_token]
                sent = [configuration['output_lang'].index2char[letter.item()] for letter in output[i] if letter not in ignore]
                y = [configuration['output_lang'].index2char[letter.item()] for letter in batch_y[i] if letter not in ignore]
                if sent == y:
                    correct += 1
                total += 1

    return (correct/total)*100

def train_with_attn(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length, teacher_forcing_ratio = 0.5):
    
    batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
    encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

    input_tensor = input_tensor.transpose(0, 1)
    target_tensor = target_tensor.transpose(0, 1)

    if configuration["cell_type"] == "LSTM":
        encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], batch_size, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]

    decoder_input = torch.LongTensor([SOS_token]*batch_size)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True 
    else:
        use_teacher_forcing = False

    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size))
            loss += criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i]

    else:
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, batch_size,decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size))
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_tensor[i])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate_with_attn(encoder, decoder, loader, configuration, criterion , max_length):

    with torch.no_grad():

        total = 0
        correct = 0
        
        for batch_x, batch_y in loader:

            batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
            encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

            input_variable = batch_x.transpose(0, 1)
            target_variable = batch_y.transpose(0, 1)
            
            if configuration["cell_type"] == "LSTM":
                encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

            input_length = input_variable.size()[0]
            target_length = target_variable.size()[0]

            output = torch.LongTensor(target_length, batch_size)

            encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size)
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
            
            for i in range(input_length):
                encoder_output, encoder_hidden = encoder(input_variable[i], batch_size, encoder_hidden)

            decoder_input = torch.LongTensor([SOS_token] * batch_size)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_hidden = encoder_hidden

            for i in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, batch_size, decoder_hidden,encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size))
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi
                output[i] = torch.cat(tuple(topi))

            output = output.transpose(0,1)
            for i in range(output.size()[0]):
                ignore = [SOS_token, EOS_token, PAD_token]
                sent = [configuration['output_lang'].index2char[letter.item()] for letter in output[i] if letter not in ignore]
                y = [configuration['output_lang'].index2char[letter.item()] for letter in batch_y[i] if letter not in ignore]
                if sent == y:
                    correct += 1
                total += 1

    return (correct/total)*100

def cal_val_loss_with_attn(encoder, decoder, input_tensor, target_tensor, configuration, criterion , max_length):

    with torch.no_grad():

        batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
        encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

        input_tensor = input_tensor.transpose(0, 1)
        target_tensor = target_tensor.transpose(0, 1)
            
        if configuration["cell_type"] == "LSTM":
            encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]

        encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size)
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0
            
        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i], batch_size, encoder_hidden)

        decoder_input = torch.LongTensor([SOS_token] * batch_size)
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size))
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output, target_tensor[i])

    return loss.item() / target_length

def trainIters(encoder, decoder, train_loader, val_loader, test_loader, learning_rate, configuration, wandb_flag):

    max_length = configuration['max_length_word']

    encoder_optimizer, decoder_optimizer = None, None

    if configuration['optimizer']=='nadam':
        encoder_optimizer = optim.NAdam(encoder.parameters(),lr=learning_rate)
        decoder_optimizer = optim.NAdam(decoder.parameters(),lr=learning_rate)
    else:
        encoder_optimizer = optim.Adam(encoder.parameters(),lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(),lr=learning_rate)

    criterion = nn.NLLLoss()
    
    ep = configuration['epochs']

    for i in range(ep):

        train_loss_total = 0
        val_loss_total = 0

        for batchx, batchy in train_loader:
            loss = None

            if configuration['attention'] == False:
                loss = train(batchx, batchy, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length)
            else:
                loss = train_with_attn(batchx, batchy, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length + 1)
            
            train_loss_total += loss
        
        train_loss_total = train_loss_total/len(train_loader)
        print('ep : ', i, ' | ', end='')
        print('train loss :', train_loss_total, ' | ', end='')

        for batchx, batchy in val_loader:
            loss = None

            if configuration['attention'] == False:
                loss = cal_val_loss(encoder, decoder, batchx, batchy, configuration, criterion , max_length)
            else:
                loss = cal_val_loss_with_attn(encoder, decoder, batchx, batchy, configuration, criterion , max_length+1)
            
            val_loss_total += loss

        val_loss_total = val_loss_total/len(val_loader)

        # train_acc = 0
        val_acc = 0
        
        # if configuration['attention'] == False:
        #     train_acc = evaluate(encoder, decoder, train_loader, configuration, criterion, max_length)
        # else:
        #     train_acc = evaluate_with_attn(encoder, decoder, train_loader, configuration, criterion, max_length+1)

        if configuration['attention'] == False:
            val_acc = evaluate(encoder, decoder, val_loader, configuration, criterion, max_length)
        else:
            val_acc = evaluate_with_attn(encoder, decoder, val_loader, configuration, criterion, max_length+1)
        
        # print("train accuracy : " ,train_acc, ' | ', end='')
        print('val loss :', val_loss_total, ' | ', end='')
        print("val accuracy : " ,val_acc)

        if wandb_flag == True:
            wandb.log({
                "train_loss"           : train_loss_total,
                "validation_loss"      : val_loss_total,
                # "train_accuracy"       : train_acc,
                "validation_accuracy"  : val_acc
                })

    temp = configuration['batch_size']
    configuration['batch_size'] = 1
    if configuration['attention'] == False:
        print("test accuracy for the model : " ,evaluate(encoder, decoder, test_loader, configuration, criterion, max_length))
    else:
        print("test accuracy for the model : " ,evaluate_with_attn(encoder, decoder, test_loader, configuration, criterion, max_length+1))
    configuration['batch_size'] = temp


def main(config = None):

    with wandb.init(config = config, entity = entity_name_ap) as run:
        
        config = wandb.config
        run.name = 'hs_'+str(config.hidden_size)+'_bs_'+str(config.batch_size)+'_ct_'+config.cell_type+'_es_'+str(config.embedding_size)+'_do_'+str(config.drop_out)+'_nle_'+str(config.num_layers_en)+'_nld_'+str(config.num_layers_dec)+'_lr_'+str(config.learning_rate)+'_bd_'+str(config.bidirectional)

        configuration = {

                'hidden_size'         : config.hidden_size,
                'source_lang'         : input_lang_ap,
                'target_lang'         : target_lang_ap,
                'cell_type'           : config.cell_type,
                'num_layers_encoder'  : config.num_layers_en,
                'num_layers_decoder'  : config.num_layers_en,
                'drop_out'            : config.drop_out, 
                'embedding_size'      : config.embedding_size,
                'bi_directional'      : config.bidirectional,
                'batch_size'          : config.batch_size,
                'attention'           : attention_ap,
                'epochs'              : config.epochs,
                'optimizer'           : config.optimizer

                # 'hidden_size'         : hidden_size_ap,
                # 'source_lang'         : input_lang_ap,
                # 'target_lang'         : target_lang_ap,
                # 'cell_type'           : cell_type_ap,
                # 'num_layers_encoder'  : num_layers_en_ap,
                # 'num_layers_decoder'  : num_layers_dec_ap,
                # 'drop_out'            : drop_out_ap, 
                # 'embedding_size'      : embedding_size_ap,
                # 'bi_directional'      : bidirectional_ap,
                # 'batch_size'          : batch_size_ap,
                # 'attention'           : attention_ap
            }
        
        
        train_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_train.csv')
        validation_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_valid.csv')
        test_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_test.csv')

        train_prepared_data= prepareData(configuration['source_lang'], configuration['target_lang'],train_path)

        input_lang = train_prepared_data['input_lang']
        output_lang = train_prepared_data['output_lang']
        pairs = train_prepared_data['pairs']
        max_input_length = train_prepared_data['max_input_length']
        max_target_length = train_prepared_data['max_target_length']
        
        val_prepared_data= prepareData(configuration['source_lang'], configuration['target_lang'],validation_path)

        val_pairs = val_prepared_data['pairs']
        max_input_length_val = val_prepared_data['max_input_length']
        max_target_length_val = val_prepared_data['max_target_length']

        test_prepared_data= prepareData(configuration['source_lang'], configuration['target_lang'],test_path)

        test_pairs = test_prepared_data['pairs']
        max_input_length_test = test_prepared_data['max_input_length']
        max_target_length_test = test_prepared_data['max_target_length']

        max_list = [max_input_length, max_target_length, max_input_length_val, max_target_length_val, max_input_length_test, max_target_length_test]
        max_len_all = max(max_list)

        max_len = max(max_input_length, max_target_length) + 2

        configuration['input_lang'] = input_lang
        configuration['output_lang'] = output_lang
        configuration['max_length_word'] = max_len_all + 1

        print(configuration['max_length_word'])

        encoder1 = EncoderRNN(input_lang.n_chars, configuration)
        decoder1 = DecoderRNN(configuration, output_lang.n_chars)
        attndecoder1 = DecoderAttention(configuration, output_lang.n_chars)
        if use_cuda:
            encoder1=encoder1.cuda()
            decoder1=decoder1.cuda()
            attndecoder1 = attndecoder1.cuda()

        pairs = tensorFromPairs(configuration['input_lang'], configuration['output_lang'], pairs , configuration['max_length_word'])
        val_pairs = tensorFromPairs(configuration['input_lang'], configuration['output_lang'], val_pairs, configuration['max_length_word'])
        test_pairs = tensorFromPairs(configuration['input_lang'], configuration['output_lang'], test_pairs, configuration['max_length_word'])

        train_loader = torch.utils.data.DataLoader(pairs, batch_size=configuration['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_pairs, batch_size=configuration['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_pairs, batch_size=1, shuffle=True)

        if configuration['attention'] == False :
            trainIters(encoder1, decoder1, train_loader, val_loader, test_loader, config.learning_rate, configuration,True)
        else : 
            trainIters(encoder1, attndecoder1, train_loader, val_loader, test_loader, config.learning_rate, configuration,True)

# main()

# wandb.agent('1j0tkjlp', main, count  = 10)

def bestRun():
        configuration = {

                'hidden_size'         : hidden_size_ap,
                'source_lang'         : input_lang_ap,
                'target_lang'         : target_lang_ap,
                'cell_type'           : cell_type_ap,
                'num_layers_encoder'  : num_layers_en_ap,
                'num_layers_decoder'  : num_layers_en_ap,
                'drop_out'            : drop_out_ap, 
                'embedding_size'      : embedding_size_ap,
                'bi_directional'      : bidirectional_ap,
                'batch_size'          : batch_size_ap,
                'attention'           : attention_ap,
                'learning_rate'       : learning_rate_ap,
                'optimizer'           : optimizer_ap,
                'epochs'              : epochs_ap
            }
        
        print(configuration)
        
        train_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_train.csv')
        validation_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_valid.csv')
        test_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_test.csv')

        train_prepared_data= prepareData(configuration['source_lang'], configuration['target_lang'],train_path)

        input_lang = train_prepared_data['input_lang']
        output_lang = train_prepared_data['output_lang']
        pairs = train_prepared_data['pairs']
        max_input_length = train_prepared_data['max_input_length']
        max_target_length = train_prepared_data['max_target_length']
        
        val_prepared_data= prepareData(configuration['source_lang'], configuration['target_lang'],validation_path)

        val_pairs = val_prepared_data['pairs']
        max_input_length_val = val_prepared_data['max_input_length']
        max_target_length_val = val_prepared_data['max_target_length']

        test_prepared_data= prepareData(configuration['source_lang'], configuration['target_lang'],test_path)

        test_pairs = test_prepared_data['pairs']
        max_input_length_test = test_prepared_data['max_input_length']
        max_target_length_test = test_prepared_data['max_target_length']

        max_list = [max_input_length, max_target_length, max_input_length_val, max_target_length_val, max_input_length_test, max_target_length_test]
        max_len_all = max(max_list)

        max_len = max(max_input_length, max_target_length) + 2

        configuration['input_lang'] = input_lang
        configuration['output_lang'] = output_lang
        configuration['max_length_word'] = max_len_all + 1

        print(configuration['max_length_word'])

        encoder1 = EncoderRNN(input_lang.n_chars, configuration)
        decoder1 = DecoderRNN(configuration, output_lang.n_chars)
        attndecoder1 = DecoderAttention(configuration, output_lang.n_chars)
        if use_cuda:
            encoder1=encoder1.cuda()
            decoder1=decoder1.cuda()
            attndecoder1 = attndecoder1.cuda()

        pairs = tensorFromPairs(configuration['input_lang'], configuration['output_lang'], pairs , configuration['max_length_word'])
        val_pairs = tensorFromPairs(configuration['input_lang'], configuration['output_lang'], val_pairs, configuration['max_length_word'])
        test_pairs = tensorFromPairs(configuration['input_lang'], configuration['output_lang'], test_pairs, configuration['max_length_word'])

        train_loader = torch.utils.data.DataLoader(pairs, batch_size=configuration['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_pairs, batch_size=configuration['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_pairs, batch_size=1, shuffle=True)

        if configuration['attention'] == False :
            trainIters(encoder1, decoder1, train_loader, val_loader, test_loader, configuration['learning_rate'], configuration,False)
        else : 
            trainIters(encoder1, attndecoder1, train_loader, val_loader, test_loader, configuration['learning_rate'], configuration,False)

bestRun()
