# Recurrent neural networks to build a transliteration system
Course project submission for the course CS6910: Fundamentals of Deep Learning.  
Check this link for the task description : [Problem Statement Link](https://wandb.ai/cs6910_2023/A3/reports/Assignment-3--Vmlldzo0MDU2MzQx)

## Descriptions of files

1. **model.py** : It is the main model that is trained for getting predictions for test sets. It consists of three modules/classes EncoderRNN, DecoderENN and DecoderAttention. The former two have been used in vanilla RNN Model while DecoderAttention was used along with EncoderRNN in attention RNN Model.

2. **train.py** : It is the main file that is to be run to train the given data set. By default it runs for the best hyperparameter sweep configuration. However commandline arguments have been added so as to customize parameters of the model. Sweeps have been run once in this file and now have been commented out.

3. **confusion_matrix.py** : It uses predictions_vanilla.csv and predictions_attention.csv to create confusion matrix for both the models respectively.

4. **predictions_vanilla.csv**: Contains Test input, target and predicted words for vanilla model.

5. **predictions_attention.csv**: Contains Test input, target and predicted words for attention model.

6. **MANGAL.TTF**: Support file for devnagri script. It has been used for labelling axes in confusion matrices and attention heatmaps.

## Running the code

As mentioned earlier, there are two python code files named train.py and model.py.<br>

### Running the train.py file from the terminal

1. Put train.py and model.py file in the same folder.<br>

2. Please make sure that the unzipped folder of the aksharantar_sampled dataset is present in the same directory as these python files.<br>

3. First you will need to run the model.py file as shown below in the terminal<br>
```sh
python model.py
```
4. You can now run train.py in two ways:<br>

   a. For best hyperparameter sweep configuration you can simply write this in terminal 
    ```sh
    python train.py
    ```
    
   b. For customized parameters for the model you can write the following in terminal
    ```sh
    python train.py --batch_size xx --learning_rate xx --hidden_size xx --embedding_size xx --bidirectional xx --attention xx
    ```
    Replace `xx` in above with the appropriate parameter you want to train the model with<br>
      For example: 

      ```sh
      python train.py --batch_size 64 --learning_rate 1e-3 --hidden_size 256 --embedding_size 256 --bidirectional True --attention False
      ```
      
### Running the train.py file from the terminal for hyperparameter sweeps
1. Put train.py and model.py file in the same folder.<br>

2. Please make sure that the unzipped folder of the aksharantar_sampled dataset is present in the same directory as these python files.<br>

3. First you will need to run the model.py file as shown below in the terminal<br>
```sh
python model.py
```

4. You need to uncomment line numbers 162 and 1154 from train.py for running sweeps. Sweep project and entity can be changed using command line arguments as:
```sh
python train.py --wandb_project your_proj_name --wandb_entity your_wandb_entity
```

#### Description of various command line arguments

1.  `-wp/--wandb_project`   : custom project name with default set to 'CS6910_A3_' (string type)
2.  `-we/--wandb_entity`    : custom entity name with default set to 'cs22m064' (string type)
3.  `-e/--epochs`           : custom number of epochs with available choices 5 or 10 and default set to 10 (integer type)
4.  `-b/--batch_size`       : custom batch size with available choices 32, 64 or 128 and default set to 128 (integer type)
5.  `-o/--optimizer`        : custom optimizer with available choices adam or nadam and default set to nadam (string type)
6.  `-lr/--learning_rate`   : custom learning rate with available choices 1e-2 or 1e-3 and default set to 1e-3 (integer type)
7.  `-nle/--num_layers_en`  : custom number layers encoder with available choices 1, 2 or 3 and default set to 3 (integer type)
8.  `-nld/--num_layers_dec` : custom number layers decoder with available choices 1, 2 or 3 and default set to 2 (integer type)
9.  `-sz/--hidden_size`     : custom hidden size with available choices 128, 256 or 512 and default set to 512 (integer type)
10. `-il/--input_lang`      : custom input language with available choices eng and default set to eng (string type)
11. `-tl/--target_lang`     : custom target language with available choices hin or tel and default set to hin (string type)
12. `-ct/--cell_type`       : custom cell type with available choices LSTM, GRU or RNN and default set to LSTM (string type)
13. `-do/--drop_out`        : custom dropout with available choices 0.0, 0.2 or 0.3 and default set to 0.0 (float type)
14. `-es/--embedding_size`  : custom embedding size with available choices 64,128 or 256 and default set to 256 (integer type)
15. `-bd/--bidirectional`   : custom bidirectional setting with available True or False and default set to True (boolean type)
16. `-at/--attention`       : custom attention setting with available choices True or False and default set to False (boolean type)

### Running the confusion_matrix.py file from the terminal
In order to run this file make sure the predictions_vanilla.csv and predictions_attention.csv are in the same directory as this one.<br>
Now the file can be simply run by writing this in terminal<br>
```sh
python confusion_matrix.py
```
