'''
When the code is just run without setting any commandline arguments explicitly it gives the train data loss, validation data loss and 
validation data accuracy for each epoch and test accuracy for the best set of hyperparameters.
'''
import os
import wandb
import csv
import torch
import random
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import EncoderRNN
from model import DecoderAttention
from model import DecoderRNN
from matplotlib.font_manager import FontProperties

'''
ARGPARSE SECTION

Commandline arguments have been added to the code using argparse. These include :

    1.  -wp/--wandb_project   : custom project name with default set to 'CS6910_A3_' (string type)
    2.  -we/--wandb_entity    : custom entity name with default set to 'cs22m064' (string type)
    3.  -e/--epochs           : custom number of epochs with available choices 5 or 10 and default set to 10 (integer type)
    4.  -b/--batch_size       : custom batch size with available choices 32, 64 or 128 and default set to 128 (integer type)
    5.  -o/--optimizer        : custom optimizer with available choices adam or nadam and default set to nadam (string type)
    6.  -lr/--learning_rate   : custom learning rate with available choices 1e-2 or 1e-3 and default set to 1e-3 (integer type)
    7.  -nle/--num_layers_en  : custom number layers encoder with available choices 1, 2 or 3 and default set to 3 (integer type)
    8.  -nld/--num_layers_dec : custom number layers decoder with available choices 1, 2 or 3 and default set to 2 (integer type)
    9.  -sz/--hidden_size     : custom hidden size with available choices 128, 256 or 512 and default set to 512 (integer type)
    10. -il/--input_lang      : custom input language with available choices eng and default set to eng (string type)
    11. -tl/--target_lang     : custom target language with available choices hin or tel and default set to hin (string type)
    12. -ct/--cell_type       : custom cell type with available choices LSTM, GRU or RNN and default set to LSTM (string type)
    13. -do/--drop_out        : custom dropout with available choices 0.0, 0.2 or 0.3 and default set to 0.0 (float type)
    14. -es/--embedding_size  : custom embedding size with available choices 64,128 or 256 and default set to 256 (integer type)
    15. -bd/--bidirectional   : custom bidirectional setting with available True or False and default set to True (boolean type)
    16. -at/--attention       : custom attention setting with available choices True or False and default set to False (boolean type)

Note : Default values have been set to the best hyperparameter configurations as obtained from running sweeps over a wide range of 
       values possible for hyperparameters. 

'''

parser=argparse.ArgumentParser()

parser.add_argument('-wp',      '--wandb_project',      help='project name',                                                    type=str,       default='A3')
parser.add_argument('-we',      '--wandb_entity',       help='entity name',                                                     type=str,       default='cs22m064'  )
parser.add_argument('-e',       '--epochs',             help='epochs',                          choices=[5,10],                 type=int,       default=10           )
parser.add_argument('-b',       '--batch_size',         help='batch sizes',                     choices=[32,64,128],            type=int,       default=128         )
parser.add_argument('-o',       '--optimizer',          help='optimizer',                       choices=['adam','nadam'],       type=str,       default='nadam'     )
parser.add_argument('-lr',      '--learning_rate',      help='learning rates',                  choices=[1e-2,1e-3],            type=float,     default=1e-3        )
parser.add_argument('-nle',     '--num_layers_en',      help='number of layers in encoder',     choices=[1,2,3],                type=int,       default=2           )
parser.add_argument('-nld',     '--num_layers_dec',     help='number of layers in decoder',     choices=[1,2,3],                type=int,       default=3           )
parser.add_argument('-sz',      '--hidden_size',        help='hidden layer size',               choices=[128,256,512],          type=int,       default=512         )
parser.add_argument('-il',      '--input_lang',         help='input language',                  choices=['eng'],                type=str,       default='eng'       )
parser.add_argument('-tl',      '--target_lang',        help='target language',                 choices=['hin','tel'],          type=str,       default='hin'       )
parser.add_argument('-ct',      '--cell_type',          help='cell type',                       choices=['LSTM','GRU','RNN'],   type=str,       default='LSTM'      )
parser.add_argument('-do',      '--drop_out',           help='drop out',                        choices=[0.0,0.2,0.3],          type=float,     default=0.2         )
parser.add_argument('-es',      '--embedding_size',     help='embedding size',                  choices=[64,128,256],           type=int,       default=128         )
parser.add_argument('-bd',      '--bidirectional',      help='bidirectional',                   choices=[True,False],           type=bool,      default=False       )
parser.add_argument('-at',      '--attention',          help='attention',                       choices=[True,False],           type=bool,      default=True        )

args=parser.parse_args()

# All the values fetched from command line are stored in variables
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

# Dataset directory has been set into the variable dir
dir = 'aksharantar_sampled'

# To run the program on cuda we first check if it's available or not
use_cuda = torch.cuda.is_available()
print(use_cuda)

#Globally declared indexes for start of word, end of word, unknown and padding tokens
SOW_token = 0
EOW_token = 1

'''
Note : The unknown token is necessary as the vocabulary is created using only the train data, if there is a character in the 
validation or test words that is not present in the above created vocabulary then it is replaced by unknown character.
'''

UNK_token = 3
PAD_token = 4

# -------------------------------------------------------------------------------------------------------------------------------------

'''
SWEEP CONFIGURATION SECTION

Helps set the configuration used during running of hyperparameter sweeps
'''

# The method used is bayes, we can also use grid or random for the purpose
sweep_config ={
    'method':'bayes'
}

# The metric is to maximize the validation accuracy
metric = {
    'name' : 'validation_accuracy',
    'goal' : 'maximize'
}
sweep_config['metric'] = metric

# Different values of hyperparameters through which I am going to sweep are set here
parameters_dict={
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

# The line below can be uncommented, to run sweeps. It assigns a sweep id for the sweep that is going to be run
# sweep_id = wandb.sweep(sweep_config, project=project_name_ap)

# -------------------------------------------------------------------------------------------------------------------------------------

'''
VOCABULARY PREPARATION SECTION

The prepareVocabulary Class helps to generate dictionaries for mapping character to index and then index back to the charcter from the vocabulary stored
This class is inspired from the Lang class present in the blog : https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html suggested by you
'''
class prepareVocabulary:
    def __init__(self, name):
        # Maps character to index
        self.char2index = {}
        # Count of characters in vocabulary. Initially there will be 4 characters (SOW,EOW,UNK,PAD) but later on others will be added
        self.n_chars = 4
        # Maps index back to character
        self.index2char = {0: '(', 1: ')',2 : '?', 3:'_'}
        self.name = name

    # A function that takes in a word and adds all its characters to the vocabulary
    def addWord(self, word):
        for i in range(len(word)):
            # This if statement ensures that characters in vocabulary are unique
            if word[i] not in self.char2index:
                self.char2index[word[i]] = self.n_chars
                self.index2char[self.n_chars] = word[i]
                self.n_chars += 1        

# -------------------------------------------------------------------------------------------------------------------------------------

'''
DATA PREPARATION SECTION

The prepareData function takes in a directory and returns the pairs of input,target for entire set of data
The prepareData function also makes use of the prepareVocabulary class objects to create hindi and english (flexible to take any two languages) vocabularies.
The prepareData function also finds maximum length words in both input and target language words and returns them which will be helpful later on
'''

'''
INPUTS TO THE FUNCTION:
    1.lang1 : input language
    2.lang2 : target language
    3. dir : path of csv file
'''

def prepareData(lang1, lang2, dir):

    # Read a CSV using the directory passed into the function
    data = pd.read_csv(dir,sep=",",names=['input', 'target'])

    # Find maximum length words in input and target language words
    max_input_length, max_target_length = max([len(inp) for inp in data['input'].to_list()]), max([len(tar) for tar in data['target'].to_list()])

    # Creating objects of prepareVocabulary class for input and target language
    # Initially objects are empty. They will be populated by the addWord function of the prepareVocabulary class
    input_lang, output_lang = prepareVocabulary(lang1),prepareVocabulary(lang2)

    # Creating pairs from the data extracted from CSV file above
    pairs = []
    input_list,target_list = data['input'].to_list(),data['target'].to_list()
    for i in range(len(input_list)):
        one_pair = [input_list[i],target_list[i]]
        pairs.append(one_pair)

    # Calling the addWord function of prepareVocabulary class for every pair of pairs created above to get the final vocabulary in both languages
    for i in range(len(pairs)):
        inp_l,out_l = pairs[i][0],pairs[i][1]
        input_lang.addWord(inp_l)
        output_lang.addWord(out_l)

    # This is dictionary contains pairs created earlier, vocabulary and maximum length of words for input and target languages
    # This is returned by the function
    prepared_data = {
        'input_lang' : input_lang,
        'output_lang' : output_lang,
        'pairs' : pairs,
        'max_input_length' : max_input_length,
        'max_target_length' : max_target_length
    }

    return prepared_data

# -------------------------------------------------------------------------------------------------------------------------------------

'''
PREPARE TENSORS SECTION

The functions under this section help in creating tensors that are going to be actually passed into the model created 
'''

'''
INPUTS TO THE FUNCTION:
    1. lang : input/output vocabulary is passed
    2. word: word whose tensor is to be formed 
    3. max_length : maximum out of length of all words from input and target words
'''
# This function will be called iteratively for each word in pairs of data and create a tensor from them.
def tensorFromWord(lang, word, max_length):
    index_list = []
    for i in range(len(word)):
        # If character is in the vocabulary then just append the index corresponding to it else append the UNK token
        if word[i] in lang.char2index.keys():
            index_list.append(lang.char2index[word[i]])
        else:
            index_list.append(UNK_token)

    indexes = index_list

    # After all indices have been appended, I appended a end of word token
    indexes.append(EOW_token)

    # To make all words of same length after end of word token I appended padding
    # Find the length of padding to add, create the list of padding using this length and append it to indices
    len_to_add = (max_length - len(indexes))
    list_pad_tokens = [PAD_token] * len_to_add
    indexes.extend(list_pad_tokens)
    # Convert the list of indexes into a tensor
    result = torch.LongTensor(indexes)
    # Shift the tensor to cuda if it is available
    return result.cuda() if use_cuda else result

'''
INPUTS TO THE FUNCTION:
    1. input_lang : input vocabulary
    2. output_lang : output vocabulary
    3. pairs : batch of pairs of input and target words 
    4. max_length : maximum out of length of all words from input and target words
'''
# This function will be called for pairs in data and create a tensor from them.
def tensorFromPairs(input_lang, output_lang, pairs, max_length):
    res = []
    for i in range(len(pairs)):
        # Called tensorFromWord iteratively individually for input vocabulary,input words and target vocabulary,target words
        input_variable = tensorFromWord(input_lang, pairs[i][0], max_length)
        target_variable = tensorFromWord(output_lang, pairs[i][1], max_length)
        # Append these input and target tensors as a pair of tensors
        res.append((input_variable, target_variable))
    return res

'''
INPUTS TO THE FUNCTION:
    1. configuration : the parameters defined either from sweep or best set
'''
# This function converts the train, validation and test data into tensors so that model can be applied to them
# In case of training the train tensors will be useful while in case of evaluation after each epoch validation tensors
# In the end after hyperparameter tuning when we get the best model we use the test tensors to get the test accuracy
def prepareTensors(configuration):

        # Conacatenating the train path and passing it through prepareData function
        train_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_train.csv')
        train_prepared_data= prepareData(configuration['source_lang'], configuration['target_lang'],train_path)    

        # Conacatenating the validation path and passing it through prepareData function
        validation_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_valid.csv')
        val_prepared_data= prepareData(configuration['source_lang'], configuration['target_lang'],validation_path)

        # Conacatenating the test path and passing it through prepareData function
        test_path = os.path.join(dir, configuration['target_lang'], configuration['target_lang'] + '_test.csv')
        test_prepared_data= prepareData(configuration['source_lang'], configuration['target_lang'],test_path)

        # Set the main input vocabulary and target vocabulary to the one recieved from the 'TRAIN' prepared data 
        # Can't look into the validation and test vocabularies
        input_lang = train_prepared_data['input_lang']
        output_lang = train_prepared_data['output_lang']

        # Set pairs of words fetched from prepareData
        # These pairs will be passed later on into tensorFromPairs to get tensors
        pairs = train_prepared_data['pairs']
        val_pairs = val_prepared_data['pairs']
        test_pairs = test_prepared_data['pairs']

        # Set max length of words fetched from prepareData
        max_input_length = train_prepared_data['max_input_length']
        max_target_length = train_prepared_data['max_target_length']
        max_input_length_val = val_prepared_data['max_input_length']
        max_target_length_val = val_prepared_data['max_target_length']
        max_input_length_test = test_prepared_data['max_input_length']
        max_target_length_test = test_prepared_data['max_target_length']

        # Find maximum out of all the maximum length of words from input and target words
        max_list = [max_input_length, max_target_length, max_input_length_val, max_target_length_val, max_input_length_test, max_target_length_test]
        max_len_all = max(max_list)

        # Set the input and target vocabulary, maximum length of words (as found above) into configuration dictionary
        configuration['input_lang'] = input_lang
        configuration['output_lang'] = output_lang
        configuration['max_length_word'] = max_len_all + 1

        # Form pairs of tensors from mere pairs of words for each train, validation and test pairs
        pairs = tensorFromPairs(configuration['input_lang'], configuration['output_lang'], pairs , configuration['max_length_word'])
        val_pairs = tensorFromPairs(configuration['input_lang'], configuration['output_lang'], val_pairs, configuration['max_length_word'])
        test_pairs = tensorFromPairs(configuration['input_lang'], configuration['output_lang'], test_pairs, configuration['max_length_word'])

        return pairs,val_pairs,test_pairs,configuration   

# -------------------------------------------------------------------------------------------------------------------------------------

'''
TRAINING and EVALUATION for model with NO ATTENTION mechanism SECTION

The functions under this section help in training and evaluation of train data, evaluation of validation data without attention mechanism.
The training and evaluation functions are inspired from the blog : https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html suggested by you

''' 
'''
INPUTS TO THE FUNCTION:
    1. input_tensor : input tensors batch
    2. target_tensor : target tensors batch
    3. encoder : encoder class object
    4. decoder : decoder class object
    5. encoder_optimizer:
    6. decoder_optimizer:
    7. criterion : loss function
    8. configuration : the parameters defined either from sweep or best set
    9. max_length : maximum out of length of all words from input and target words 
'''
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length, teacher_forcing_ratio = 0.5):
    
    # Fetch batch size and number of layers from configuration dictionary
    batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
    # Initial encoder hidden is set to a tensor of zeros using initHidden function
    encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

    # Transpose the batch of input and target tensors recieved
    input_tensor = input_tensor.transpose(0, 1)
    target_tensor = target_tensor.transpose(0, 1)

    # Incase of LSTM cell the encoder hidden is a tupple of (encoder hidden, encoder cell state) while for the other two it is just encoder hidden
    if configuration["cell_type"] == "LSTM":
        encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

    # Sets the gradients of all optimized torch.Tensor s to zero.
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Finding the length of input and target tensor batches
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Initializing loss to 0
    loss = 0

    # Passing ith character of every word from the input batch into the encoder iteratively
    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], batch_size, encoder_hidden)

    # Setting a list of start of word tokens to pass into the decoder initially and converting it into tensors
    sow_list = [SOW_token]*batch_size
    decoder_input = torch.LongTensor(sow_list)
    # Shift decoder_inputs to cuda if available
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # decoder_hidden is set to the final encoder_hidden after the encoder loop completes
    decoder_hidden = encoder_hidden

    # If a random number generated is less than 0.5 then teacher forcing will be used else it won't be used
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True 
    else:
        use_teacher_forcing = False

    # In case of teacher forcing we give the next decoder input to be the ith character of every word from the target batch
    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden= decoder(decoder_input, batch_size, decoder_hidden)
            decoder_input = target_tensor[i]
            # loss being calculated from decoder ouput
            loss += criterion(decoder_output, target_tensor[i])

    # Else we give the decoder input to be the best predictions for i+1th charcter for the whole batch from ith time step
    else:
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, batch_size,decoder_hidden)
            # loss being calculated from decoder ouput
            loss += criterion(decoder_output, target_tensor[i])
            # Best prediction comes from using topk(k=1) function
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

    # Calling backward propagation
    loss.backward()

    # Update the parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return loss for the current train batch
    return loss.item() / target_length
'''
INPUTS TO THE FUNCTION:
    1. loader : validation or test set batches
    2. encoder : encoder class object
    3. decoder : decoder class object
    4. configuration : the parameters defined either from sweep or best set
    5. max_length : maximum out of length of all words from input and target words 
    6. test : if it is true then outputs will be logged into csv
'''
def evaluate(encoder, decoder, loader, configuration,max_length,test=False):

    # disabled gradient calculation for inference, helps reduce memory consumption for computations
    with torch.no_grad():

        total = 0
        correct = 0
        actual_X = []
        actual_Y = []
        predicted_Y = []
        
        # Calculating Accuracy for all batches in validation loader together
        for batch_x, batch_y in loader:
            # Fetch batch size and number of layers from configuration dictionary
            batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
            # Initial encoder hidden is set to a tensor of zeros using initHidden function
            encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

            # Transpose the batch_x and batch_y tensors 
            input_variable = batch_x.transpose(0, 1)
            target_variable = batch_y.transpose(0, 1)

            # Incase of LSTM cell the encoder hidden is a tupple of (encoder hidden, encoder cell state) while for the other two it is just encoder hidden
            if configuration["cell_type"] == "LSTM":
                encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

            # Finding the length of input and target tensor batches
            input_length = input_variable.size()[0]
            target_length = target_variable.size()[0]

            # Initializing output as a tensor of size target_length X batch_size
            output = Variable(torch.LongTensor(target_length, batch_size))

            if test:
                x = None
                for i in range(batch_x.size()[0]):
                    x = [configuration['input_lang'].index2char[letter.item()] for letter in batch_x[i] if letter not in [SOW_token, EOW_token, PAD_token, UNK_token]]
                    actual_X.append(x)
            
            # Passing ith character of every word from the input batch into the encoder iteratively  
            for i in range(input_length):
                encoder_output, encoder_hidden = encoder(input_variable[i], batch_size, encoder_hidden)

            # Setting a list of start of word tokens to pass into the decoder initially and converting it into tensors
            sow_list = [SOW_token] * batch_size
            decoder_input = torch.LongTensor(sow_list)
            # Shift decoder_inputs to cuda if available
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            # decoder_hidden is set to the final encoder_hidden after the encoder loop completes
            decoder_hidden = encoder_hidden

            # We are just evaluating the output of decoder in this case so we don't use teacher forcing here
            # We give the decoder input to be the best predictions for i+1th charcter for the whole batch from ith time step
            for i in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, batch_size, decoder_hidden)
                # Best prediction comes from using topk(k=1) function
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi
                # Storing prediction of decoder at every time step into the output tensor
                output[i] = torch.cat(tuple(topi))

            # Taking transpose of output and finding it's length
            output = output.transpose(0,1)
            output_length = output.size()[0]

            # Checking if all output words of a batch match their corresponding target word
            for i in range(output_length):
                # sent is ith predicted word of a batch
                sent = [configuration['output_lang'].index2char[letter.item()] for letter in output[i] if letter not in [SOW_token, EOW_token, PAD_token, UNK_token]]
                #y is ith target word of the batch 
                y = [configuration['output_lang'].index2char[letter.item()] for letter in batch_y[i] if letter not in [SOW_token, EOW_token, PAD_token, UNK_token]]
                if test:
                    actual_Y.append(y)
                    predicted_Y.append(sent)
                # If they match correct count increases by 1
                if sent == y:
                    correct += 1
                # Counts total number of such pairs of target and predicted words
                total += 1
        if test:
            writeToCSV(actual_X,actual_Y,predicted_Y)
    
    # Accuracy will be correct/total and multiply by 100 to get in percentage
    return (correct/total)*100
'''
INPUTS TO THE FUNCTION:
    1. input_tensor : input tensors batch
    2. target_tensor : target tensors batch
    3. encoder : encoder class object
    4. decoder : decoder class object
    5. criterion : loss function
    6. configuration : the parameters defined either from sweep or best set
    7. max_length : maximum out of length of all words from input and target words 
'''
def cal_val_loss(encoder, decoder, input_tensor, target_tensor, configuration, criterion , max_length):

    # disabled gradient calculation for inference, helps reduce memory consumption for computations
    with torch.no_grad():
        
        # Fetch batch size and number of layers from configuration dictionary
        batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
        # Initial encoder hidden is set to a tensor of zeros using initHidden function
        encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

        # Transpose the batch of input and target tensors recieved
        input_tensor = input_tensor.transpose(0, 1)
        target_tensor = target_tensor.transpose(0, 1)

        # Incase of LSTM cell the encoder hidden is a tupple of (encoder hidden, encoder cell state) while for the other two it is just encoder hidden
        if configuration["cell_type"] == "LSTM":
            encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

        # Finding the length of input and target tensor batches
        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]
        
        # Initializing loss to 0
        loss = 0

        # Passing ith character of every word from the input batch into the encoder iteratively  
        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i], batch_size, encoder_hidden)

        # Setting a list of start of word tokens to pass into the decoder initially and converting it into tensors
        sow_list = [SOW_token] * batch_size
        decoder_input = torch.LongTensor(sow_list)
        # Shift decoder_inputs to cuda if available
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        # decoder_hidden is set to the final encoder_hidden after the encoder loop completes
        decoder_hidden = encoder_hidden

        # We are just evaluating the loss in this case so we don't use teacher forcing here
        # We give the decoder input to be the best predictions for i+1th charcter for the whole batch from ith time step
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, batch_size, decoder_hidden)
            # loss being calculated from decoder ouput
            loss += criterion(decoder_output, target_tensor[i])
            # Best prediction comes from using topk(k=1) function
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

    # Return loss for the current validation batch
    return loss.item() / target_length

# -------------------------------------------------------------------------------------------------------------------------------------

'''
TRAINING and EVALUATION for model with ATTENTION mechanism SECTION

The functions under this section help in training and evaluation of train data, evaluation of validation data using attention mechanism.
The training and evaluation functions are inspired from the blog : https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html suggested by you

''' 
'''
INPUTS TO THE FUNCTION:
    1. input_tensor : input tensors batch
    2. target_tensor : target tensors batch
    3. encoder : encoder class object
    4. decoder : decoder class object
    5. encoder_optimizer:
    6. decoder_optimizer:
    7. criterion : loss function
    8. configuration : the parameters defined either from sweep or best set
    9. max_length : maximum out of length of all words from input and target words 
'''
def train_with_attn(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length, teacher_forcing_ratio = 0.5):
    
    # Fetch batch size and number of layers from configuration dictionary
    batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
    # Initial encoder hidden is set to a tensor of zeros using initHidden function
    encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

    # Transpose the batch of input and target tensors recieved
    input_tensor = input_tensor.transpose(0, 1)
    target_tensor = target_tensor.transpose(0, 1)

    # Incase of LSTM cell the encoder hidden is a tupple of (encoder hidden, encoder cell state) while for the other two it is just encoder hidden
    if configuration["cell_type"] == "LSTM":
        encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

    # Sets the gradients of all optimized torch.Tensor s to zero.
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Finding the length of input and target tensor batches
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Initiaize loss to 0
    loss = 0

    # In attention mechanism we need encoder ouputs at every time step to be stored and so we will initialize a tensor that will be used to store them
    encoder_outputs = None
    if configuration['bi_directional'] :
        # Incase of bidirectional hidden size needs to be doubled
        encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size*2)
        # Shift encoder outputs to cuda if available
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    else:
        encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size)
        # Shift encoder outputs to cuda if available
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # Passing ith character of every word from the input batch into the encoder iteratively
    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], batch_size, encoder_hidden)
        encoder_outputs[i] = encoder_output[0]

    # Setting a list of start of word tokens to pass into the decoder initially and converting it into tensors
    sow_list = [SOW_token] * batch_size
    decoder_input = torch.LongTensor(sow_list)
    # Shift decoder_inputs to cuda if available
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # decoder_hidden is set to the final encoder_hidden after the encoder loop completes
    decoder_hidden = encoder_hidden

    # If a random number generated is less than 0.5 then teacher forcing will be used else it won't be used
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True 
    else:
        use_teacher_forcing = False

    # Initialize them before using them
    decoder_output, decoder_attention = None, None

    # In case of teacher forcing we give the next decoder input to be the ith character of every word from the target batch
    if use_teacher_forcing:
        for i in range(target_length):
            # In case of attention mechanism the attention decoder needs encoder outputs to be passed and it returns attention weights
            if configuration['bi_directional'] :
                # In case of bidirectional reshaped encoder outputs for twice the hidden size 
                decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size, max_length, encoder.hidden_size*2))
            else:
                decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size, max_length, encoder.hidden_size))
            decoder_input = target_tensor[i]
            # loss being calculated from decoder ouput
            loss += criterion(decoder_output, target_tensor[i])

    # Else we give the decoder input to be the best predictions for i+1th charcter for the whole batch from ith time step
    else:
        for i in range(target_length):
            if configuration['bi_directional'] :
                # In case of bidirectional reshaped encoder outputs for twice the hidden size 
                decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size*2))
            else:
                decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size))
            # loss being calculated from decoder ouput
            loss += criterion(decoder_output, target_tensor[i])
            # Best prediction comes from using topk(k=1) function
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

    # Calling backward propagation
    loss.backward()

    # Update the parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return loss for the current train batch
    return loss.item() / target_length

'''
INPUTS TO THE FUNCTION:
    1. loader : validation or test set batches
    2. encoder : encoder class object
    3. decoder : decoder class object
    4. configuration : the parameters defined either from sweep or best set
    5. max_length : maximum out of length of all words from input and target words 
    6. test : if it is true then outputs will be logged into csv
'''
def evaluate_with_attn(encoder, decoder, loader, configuration, max_length, test=False):

    # disabled gradient calculation for inference, helps reduce memory consumption for computations
    with torch.no_grad():

        total = 0
        correct = 0
        actual_X = []
        actual_Y = []
        predicted_Y = []
        
        # Calculating Accuracy for all batches in validation loader together
        for batch_x, batch_y in loader:
            # Fetch batch size and number of layers from configuration dictionary
            batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
            # Initial encoder hidden is set to a tensor of zeros using initHidden function
            encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

            # Transpose the batch_x and batch_y tensors 
            input_variable = batch_x.transpose(0, 1)
            target_variable = batch_y.transpose(0, 1)
            
            # Incase of LSTM cell the encoder hidden is a tupple of (encoder hidden, encoder cell state) while for the other two it is just encoder hidden
            if configuration["cell_type"] == "LSTM":
                encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

            # Finding the length of input and target tensor batches
            input_length = input_variable.size()[0]
            target_length = target_variable.size()[0]

            # Initializing output as a tensor of size target_length X batch_size
            output = torch.LongTensor(target_length, batch_size)

            # In attention mechanism we need encoder ouputs at every time step to be stored and so we will initialize a tensor that will be used to store them
            encoder_outputs = None
            if configuration['bi_directional'] :
                # Incase of bidirectional hidden size needs to be doubled
                encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size*2))
                # Shift encoder outputs to cuda if available
                encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
            else:
                encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
                # Shift encoder outputs to cuda if available
                encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
            
            if test:
                x = None
                for i in range(batch_x.size()[0]):
                    x = [configuration['input_lang'].index2char[letter.item()] for letter in batch_x[i] if letter not in [SOW_token, EOW_token, PAD_token, UNK_token]]
                    actual_X.append(x)

            # Passing ith character of every word from the input batch into the encoder iteratively  
            for i in range(input_length):
                encoder_output, encoder_hidden = encoder(input_variable[i], batch_size, encoder_hidden)
                encoder_outputs[i] = encoder_output[0]

            # Setting a list of start of word tokens to pass into the decoder initially and converting it into tensors
            sow_list = [SOW_token] * batch_size
            decoder_input = torch.LongTensor(sow_list)
            # Shift decoder_inputs to cuda if available
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            # decoder_hidden is set to the final encoder_hidden after the encoder loop completes
            decoder_hidden = encoder_hidden
            # Initialize them before using them
            decoder_output,decoder_attention = None,None

            # We are just evaluating the output of decoder in this case so we don't use teacher forcing here
            # We give the decoder input to be the best predictions for i+1th charcter for the whole batch from ith time step
            for i in range(target_length):
                if configuration['bi_directional'] :
                    # In case of bidirectional reshaped encoder outputs for twice the hidden size 
                    decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size*2))
                else:
                    decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size))
                # Best prediction comes from using topk(k=1) function
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi
                # Storing prediction of decoder at every time step into the output tensor
                output[i] = torch.cat(tuple(topi))

            # Taking transpose of output and finding it's length
            output = output.transpose(0,1)
            output_length = output.size()[0]
            # Checking if all output words of a batch match their corresponding target word
            for i in range(output_length):
                # sent is ith predicted word of a batch
                sent = [configuration['output_lang'].index2char[letter.item()] for letter in output[i] if letter not in [SOW_token, EOW_token, PAD_token, UNK_token]]
                #y is ith target word of the batch 
                y = [configuration['output_lang'].index2char[letter.item()] for letter in batch_y[i] if letter not in [SOW_token, EOW_token, PAD_token, UNK_token]]
                if test:
                    actual_Y.append(y)
                    predicted_Y.append(sent)
                # If they match correct count increases by 1
                if sent == y:
                    correct += 1
                # Counts total number of such pairs of target and predicted words
                total += 1
        if test:
            writeToCSV(actual_X,actual_Y,predicted_Y)
    # Accuracy will be correct/total and multiply by 100 to get in percentage
    return (correct/total)*100

'''
INPUTS TO THE FUNCTION:
    1. input_tensor : input tensors batch
    2. target_tensor : target tensors batch
    3. encoder : encoder class object
    4. decoder : decoder class object
    5. criterion : loss function
    6. configuration : the parameters defined either from sweep or best set
    7. max_length : maximum out of length of all words from input and target words 
'''
def cal_val_loss_with_attn(encoder, decoder, input_tensor, target_tensor, configuration, criterion , max_length):

    # disabled gradient calculation for inference, helps reduce memory consumption for computations
    with torch.no_grad():

        # Fetch batch size and number of layers from configuration dictionary
        batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
        # Initial encoder hidden is set to a tensor of zeros using initHidden function
        encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

        # Initialize loss to 0
        loss = 0

        # Transpose the batch of input and target tensors recieved
        input_tensor = input_tensor.transpose(0, 1)
        target_tensor = target_tensor.transpose(0, 1)

        # Incase of LSTM cell the encoder hidden is a tupple of (encoder hidden, encoder cell state) while for the other two it is just encoder hidden
        if configuration["cell_type"] == "LSTM":
            encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

        # Finding the length of input and target tensor batches
        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]

        encoder_outputs = None
        if configuration['bi_directional'] :
            encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size*2))
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        else:
            encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        # Passing ith character of every word from the input batch into the encoder iteratively  
        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i], batch_size, encoder_hidden)
            encoder_outputs[i] = encoder_output[0]

        # Setting a list of start of word tokens to pass into the decoder initially and converting it into tensors
        sow_list = [SOW_token] * batch_size
        decoder_input = torch.LongTensor(sow_list)
        # Shift decoder_inputs to cuda if available
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        # decoder_hidden is set to the final encoder_hidden after the encoder loop completes
        decoder_hidden = encoder_hidden
        decoder_output,decoder_attention = None,None

        # We are just evaluating the loss in this case so we don't use teacher forcing here
        # We give the decoder input to be the best predictions for i+1th charcter for the whole batch from ith time step
        for i in range(target_length):
            if configuration['bi_directional'] :
                decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size*2))
            else:
                decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size))
            # loss being calculated from decoder ouput
            loss += criterion(decoder_output, target_tensor[i])
            # Best prediction comes from using topk(k=1) function
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Return loss for the current validation batch
    return loss.item() / target_length

# -------------------------------------------------------------------------------------------------------------------------------------

'''
HEATMAPS SECTION

The functions under this section help in creating and plotting heatmaps.
''' 

'''
INPUTS TO THE FUNCTION:
    1. loader : test set batches
    2. encoder : encoder class object
    3. decoder : decoder class object
    4. configuration : the parameters defined either from sweep or best set
    5. max_length : maximum out of length of all words from input and target words 
'''
def store_heatmaps(encoder, decoder, loader, configuration, max_length):

    temp = configuration['batch_size']
    # Evaluating for 10 test inputs so batch size is set to 1
    configuration['batch_size'] = 1

    # disabled gradient calculation for inference, helps reduce memory consumption for computations
    with torch.no_grad():

        # Need heatmaps for 10 inputs only that will be ensured by count
        count = 0
        # Need to store predicted y's, x's and attentions corresponding to these 10 inputs
        predictions = []
        xs = []
        attentions = []

        # Loops until 10 points have been covered
        for batch_x, batch_y in loader:
            count+=1
            # Fetch batch size and number of layers from configuration dictionary
            batch_size,num_layers_enc = configuration['batch_size'],configuration['num_layers_encoder']
            # Initial encoder hidden is set to a tensor of zeros using initHidden function
            encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

            decoder_attentions = torch.zeros(max_length, batch_size, max_length)

            # Transpose the batch_x and batch_y tensors 
            input_variable = batch_x.transpose(0, 1)
            target_variable = batch_y.transpose(0, 1)

            # Incase of LSTM cell the encoder hidden is a tupple of (encoder hidden, encoder cell state) while for the other two it is just encoder hidden
            if configuration["cell_type"] == "LSTM":
                encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

            # Finding the length of input and target tensors
            input_length = input_variable.size()[0]
            target_length = target_variable.size()[0]

            # Appending the input word
            x = None
            for i in range(batch_x.size()[0]):
                x = [configuration['input_lang'].index2char[letter.item()] for letter in batch_x[i] if letter not in [SOW_token, EOW_token, PAD_token, UNK_token]]
                xs.append(x)

            # Initializing output as a tensor of size target_length X batch_size
            output = torch.LongTensor(target_length, batch_size)

            # In attention mechanism we need encoder ouputs at every time step to be stored and so we will initialize a tensor that will be used to store them
            encoder_outputs = None
            if configuration['bi_directional'] :
                # Incase of bidirectional hidden size needs to be doubled                
                encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size*2))
                # Shift encoder outputs to cuda if available
                encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
            else:
                encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
                # Shift encoder outputs to cuda if available
                encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

            # Passing ith character of every word from the input batch into the encoder iteratively  
            for i in range(input_length):
                encoder_output, encoder_hidden = encoder(input_variable[i], batch_size, encoder_hidden)
                encoder_outputs[i] = encoder_output[0]

            # Setting a list of start of word tokens to pass into the decoder initially and converting it into tensors
            sow_list = [SOW_token] * batch_size
            decoder_input = torch.LongTensor(sow_list)
            # Shift decoder_inputs to cuda if available
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            # decoder_hidden is set to the final encoder_hidden after the encoder loop completes
            decoder_hidden = encoder_hidden
            # Initialize them before using them
            decoder_output,decoder_attention = None,None

            # We are just evaluating the output of decoder in this case so we don't use teacher forcing here
            # We give the decoder input to be the best predictions for i+1th charcter for the whole batch from ith time step
            for i in range(target_length):
                if configuration['bi_directional'] :
                    # In case of bidirectional reshaped encoder outputs for twice the hidden size 
                    decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size*2))
                else:
                    decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hidden_size))
                # Best prediction comes from using topk(k=1) function                
                decoder_attentions[i] = decoder_attention.data 
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi
                output[i] = torch.cat(tuple(topi))
            
            # Appending the attentions for every input word
            attentions.append(decoder_attentions)

            # Taking transpose of output and finding it's length
            output = output.transpose(0,1)
            output_length = output.size()[0]
            for i in range(output_length):
                # sent is ith predicted word of a batch
                sent = [configuration['output_lang'].index2char[letter.item()] for letter in output[i] if letter not in [SOW_token, EOW_token, PAD_token, UNK_token]]
                # Appending the predicted words for the input word
                predictions.append(sent)

            if count == 12 : 
                configuration['batch_size'] = temp
                # Returns input words, predicted words, attentions respectively
                return predictions,attentions,xs
'''
INPUTS TO THE FUNCTION:
    1. test_loader : test set batches
    2. encoder : encoder class object
    3. decoder : decoder class object
    4. configuration : the parameters defined either from sweep or best set
'''
def plot_heatmap(configuration, test_loader, encoder, decoder):
    max_length = configuration['max_length_word']
    # input words, predicted words, attentions respectively fetched from store_heatmaps
    predictions,attentions,test_english = store_heatmaps(encoder, decoder, test_loader, configuration, max_length+1)
    # fig will store the figure with 10 subplots
    fig = []
    n = 12 
    fig , axs = plt.subplots(4,3)
    fig.set_size_inches(23, 15)
    l = -1
    k = 0
    # Iterate 12 times
    for i in range(n):
        attn_weights = []
        # Fetch attention corresponding to ith input word
        attn_weight = attentions[i].reshape(-1,max_length+1)
        ylabel = [""]
        xlabel = [""]
        m = len(predictions[i])
        # ylabel will have predicted word
        ylabel += [char for char in predictions[i]]
        # xlabel will have input word
        xlabel += [char for char in test_english[i]]
        
        # y will be of size of ylable
        for j in range(1,m+1):
            # x will be of size of xlabel
            fg = attn_weight[j][1:len(xlabel)]
            attn_weights.append(fg.numpy())
            
        attn_weights = attn_weights[:-1]
        # After every 3 goto next line
        if i%3 == 0:
            l+=1
            k=0
        # plot the attention heatmap
        cax = axs[l][k].matshow(np.array(attn_weights))
        # set xlabels
        axs[l][k].set_xticklabels(xlabel)
        # set ylabels with support for hindi text
        xyz = FontProperties(fname = "MANGAL.TTF", size = 10)
        axs[l][k].set_yticklabels(ylabel, fontproperties = xyz)
        k+=1
    # Plot on wandb
    run = wandb.init(project=project_name_ap,entity = entity_name_ap)
    plt.show()
    wandb.log({'heatmaps':fig})
    wandb.finish()

# -------------------------------------------------------------------------------------------------------------------------------------

'''
TEST ACCURACY FOR BEST MODEL SECTION

The functions under this section help in printing test accuracy for best model.
''' 
'''
INPUTS TO THE FUNCTION:
    1. test_loader : test set batches
    2. encoder : encoder class object
    3. decoder : decoder class object
    4. configuration : the parameters defined either from sweep or best set
'''
def test_acc_best_model(configuration,test_loader,encoder,decoder):

    max_length = configuration['max_length_word']
    temp = configuration['batch_size']
    # Making batch size 1 to evaluate for just one input as a batch 
    configuration['batch_size'] = 1
    if not configuration['attention']:
        # In case of non-attention calling evaluate()
        print("test accuracy for the model : " ,evaluate(encoder, decoder, test_loader, configuration, max_length,True))
    else:
        # Else calling evaluate_with_attn()
        print("test accuracy for the model : " ,evaluate_with_attn(encoder, decoder, test_loader, configuration, max_length+1,True))
    # Returning batch size back to previous
    configuration['batch_size'] = temp

# -------------------------------------------------------------------------------------------------------------------------------------

'''
TRAINING OF THE MODEL SECTION

The functions under this section help in training the model.
''' 
'''
INPUTS TO THE FUNCTION:
    1. train_loader : train set batches
    2. val_loader : validation set batches
    3. encoder : encoder class object
    4. decoder : decoder class object
    5. configuration : the parameters defined either from sweep or best set
    6. wand_flag : if code runs in sweep mode it is true else false
'''
def trainModel(encoder, decoder, train_loader, val_loader, configuration, wandb_flag):
    
    # Using cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # Set parameters from configuration dictionary
    learning_rate = configuration['learning_rate']
    max_length = configuration['max_length_word']
    ep = configuration['epochs']

    # Set optimizer from configuration dictionary
    encoder_optimizer, decoder_optimizer = None, None

    if configuration['optimizer']=='nadam':
        encoder_optimizer = optim.NAdam(encoder.parameters(),lr=learning_rate)
        decoder_optimizer = optim.NAdam(decoder.parameters(),lr=learning_rate)
    else:
        encoder_optimizer = optim.Adam(encoder.parameters(),lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(),lr=learning_rate)

    # Iterate over epochs
    for i in range(ep):

        train_loss_total = 0
        val_loss_total = 0

        loss = None

        # Calculate train loss batchwise
        for batchx, batchy in train_loader:

            if not configuration['attention']:
                loss = train(batchx, batchy, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length)
            else:
                loss = train_with_attn(batchx, batchy, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length + 1)
            # Add train loss for all batches
            train_loss_total += loss
        
        #Divide aggregated train loss by number of batches  
        train_loss_total = train_loss_total/len(train_loader)
        print('ep : ', i, ' | ', end='')
        print('train loss :', train_loss_total, ' | ', end='')

        loss = None
        
        # Calculate validation loss batchwise
        for batchx, batchy in val_loader:

            if not configuration['attention']:
                loss = cal_val_loss(encoder, decoder, batchx, batchy, configuration, criterion , max_length)
            else:
                loss = cal_val_loss_with_attn(encoder, decoder, batchx, batchy, configuration, criterion , max_length+1)
            # Add validation loss for all batches
            val_loss_total += loss

        #Divide aggregated validation loss by number of batches  
        val_loss_total = val_loss_total/len(val_loader)

        # I commented out the code for train accuracy as it was time taking in sweeps but you can uncomment and test it

        # train_acc = 0
        
        # if configuration['attention'] == False:
        #     train_acc = evaluate(encoder, decoder, train_loader, configuration, max_length)
        # else:
        #     train_acc = evaluate_with_attn(encoder, decoder, train_loader, configuration, max_length+1)

        val_acc = 0

        # Calculated Validatiion Accuracy
        if configuration['attention'] == False:
            val_acc = evaluate(encoder, decoder, val_loader, configuration,max_length)
        else:
            val_acc = evaluate_with_attn(encoder, decoder, val_loader, configuration,max_length+1)
        
        # print("train accuracy : " ,train_acc, ' | ', end='')
        print('val loss :', val_loss_total, ' | ', end='')
        print("val accuracy : " ,val_acc)

        # If wandb flag is true then train loss, train accuracy, validation loss, validation accuracy are logged into wandb
        if wandb_flag == True:
            wandb.log({
                "train_loss"           : train_loss_total,
                "validation_loss"      : val_loss_total,
                # "train_accuracy"       : train_acc,
                "validation_accuracy"  : val_acc
                })

# -------------------------------------------------------------------------------------------------------------------------------------

'''
WRITE TO CSV SECTION

The functions under this section help in writing the test predictions to csv.
'''
'''
INPUTS TO THE FUNCTION:
    1. actual_input_list : test set input word list
    2. actual_targtet_list : test set target word list
    3. predicted_target_list : test set predictions
'''
def writeToCSV(actual_input_list,actual_targtet_list,predicted_target_list):

    rows = []
    for i in range(len(predicted_target_list)):
        str_X = ''
        str_X = str_X.join(actual_input_list[i])
        str_Y = ''
        str_Y = str_Y.join(actual_targtet_list[i])
        str_pred_Y = ''
        str_pred_Y = str_pred_Y.join(predicted_target_list[i])
        rows.append([str_X,str_Y,str_pred_Y])
    
    filename = "predictions_attention.csv"
    fields = ["Actual_X","Actual_Y","Predicted_Y"]

    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)


# -------------------------------------------------------------------------------------------------------------------------------------

'''
SWEEP RUN AND BEST RUN SECTION

The functions under this section help in running sweeps.
'''

def sweepRun(config = None):

    # Calling wandb.init() that initializes a run
    with wandb.init(config = config, entity = entity_name_ap) as run:
        
        # Storing wandb configurations
        config = wandb.config
        # Setting run name related to hyperparameter set
        run.name = 'hs_'+str(config.hidden_size)+'_bs_'+str(config.batch_size)+'_ct_'+config.cell_type+'_es_'+str(config.embedding_size)+'_do_'+str(config.drop_out)+'_nle_'+str(config.num_layers_en)+'_nld_'+str(config.num_layers_dec)+'_lr_'+str(config.learning_rate)+'_bd_'+str(config.bidirectional)

        # Setting wandb config to local configuration dictioary
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
                'epochs'              : epochs_ap,
                'optimizer'           : config.optimizer,
                'learning_rate'       : config.learning_rate
            }

        # Preparing tensors from data 
        pairs,val_pairs,test_pairs,configur = prepareTensors(configuration)

        configuration = configur

        # Setting the input and target vocabulary recieved from configuration dictionary
        input_lang = configuration['input_lang']
        output_lang = configuration['output_lang']

        # Initializing encoder, decoder and attention decoder objects
        encoder = EncoderRNN(input_lang.n_chars, configuration)
        decoder = DecoderRNN(configuration, output_lang.n_chars)
        attndecoder = DecoderAttention(configuration, output_lang.n_chars)
        # Shifting them to cuda if available
        if use_cuda:
            encoder=encoder.cuda()
            decoder=decoder.cuda()
            attndecoder = attndecoder.cuda()

        # Using DataLoader class to make shuffled batches for train and validation dataset
        train_loader = torch.utils.data.DataLoader(pairs, batch_size=configuration['batch_size'], shuffle=True)
        wandb_flag = True
        val_loader = torch.utils.data.DataLoader(val_pairs, batch_size=configuration['batch_size'], shuffle=True)

        # Calling trainIters wrt too attention
        if not configuration['attention']:
            trainModel(encoder, decoder, train_loader, val_loader, configuration,wandb_flag)
        else : 
            trainModel(encoder, attndecoder, train_loader, val_loader, configuration,wandb_flag)

# This line below needs to be uncommented in order to run sweeps
# wandb.agent(sweep_id, sweepRun, count  = 1)

def bestRun():
        # Best parameters are set as default in arparse parameters
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

        # Preparing tensors from data 
        pairs,val_pairs,test_pairs,configur = prepareTensors(configuration)

        configuration = configur

        # Setting the input and target vocabulary recieved from configuration dictionary
        input_lang = configuration['input_lang']
        output_lang = configuration['output_lang']

        # Initializing encoder, decoder and attention decoder objects
        encoder1 = EncoderRNN(input_lang.n_chars, configuration)
        decoder1 = DecoderRNN(configuration, output_lang.n_chars)
        attndecoder1 = DecoderAttention(configuration, output_lang.n_chars)
        # Shifting them to cuda if available
        if use_cuda:
            encoder1=encoder1.cuda()
            decoder1=decoder1.cuda()
            attndecoder1 = attndecoder1.cuda()

        # Using DataLoader class to make shuffled batches for train and validation dataset
        train_loader = torch.utils.data.DataLoader(pairs, batch_size=configuration['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_pairs, batch_size=1, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_pairs, batch_size=configuration['batch_size'], shuffle=True)
        
        wandb_flag = False

        # Calling trainIters wrt too attention
        if not configuration['attention'] :
            trainModel(encoder1, decoder1, train_loader, val_loader, configuration,wandb_flag)
            # Here we also print the test accuracy for best model
            test_acc_best_model(configuration, test_loader, encoder1, decoder1)
        else : 
            trainModel(encoder1, attndecoder1, train_loader, val_loader, configuration,wandb_flag)
            # Plotting heatmaps for best model with attention
            # plot_heatmap(configuration, test_loader, encoder1, attndecoder1)
            # Here we also print the test accuracy for best model
            test_acc_best_model(configuration, test_loader, encoder1, attndecoder1)

bestRun()
