# Chatbot

* Name: KATTA 🐶
* Version: 0.1
* Framework: Tensorflow 2.7.0
* Model Architecture: Many to Many LSTM
* Dataset: [cornell movie dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
* Created by: Venkat R

## Introduction

> KATTA🐶 was created using Deep Learning with Tensorflow 2.7.0. He was trained with the [cornell movie dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). The Model architecture is based Many to Many LSTM (Encoder-Decoder). This idea is partially inspired from Suryadeepan's [practical_seq2seq](https://github.com/suriyadeepan/practical_seq2seq). 

> KATTA🐶 is currently developed as a General purpose chatbot who respond to normal chats. He doesn't serve any specific purpose as comercial chatbots.


### Usage

* The Experiments were carried out in the file [Katta_bot_experiment](https://github.com/VENKATESHWARAN-R/Chatbot/blob/master/Katta_bot_experiment.ipynb) file where the code for different set of experiments are available.
* From the different experiments carried out the best performing model weights are saved and used in the main file [katta_bot](https://github.com/VENKATESHWARAN-R/Chatbot/blob/master/katta_bot.ipynb).
* The experiments are carried out in google [colab](https://colab.research.google.com/) using the GPU.
* The data processing and batch preparing functions are seperated in [data_utils.ipynb](https://github.com/VENKATESHWARAN-R/Chatbot/blob/master/data_utils.ipynb)/[data_utils.py](https://github.com/VENKATESHWARAN-R/Chatbot/blob/master/data_utils.py) file (both are same files only the format differes)
* [katta_bot](https://github.com/VENKATESHWARAN-R/Chatbot/blob/master/katta_bot.ipynb) file is the main file contains the code with the best performing model.
* Model weights are stored seperately instead of saving the whole model. This is due to the customization options can be perfromed on the model outputs.


### Model Response

Following are some of the sample conversations had with KATTA🐶
```
Human:  Hi
KATTA: hello 
Human:  How are you
KATTA: all right 
Human:  what are you doing
KATTA: i am thinking 
Human:  good
KATTA: well i got something to do with it 
Human:  do you need any help
KATTA: i do not think i am 
Human:  are you a robot
KATTA: no 
Human:  are you a human
KATTA: yes 
Human:  do you know me
KATTA: yes 
Human:  where are you
KATTA: i am here 
Human:  who are you
KATTA: gondorff asked me to meet you 
Human:  do you want to say anything
KATTA: no 
Human:  okay great
KATTA: you are not 
Human:  do you like me
KATTA: it was not me 
Human:  good night
KATTA: good night 
Human:  what's your name
KATTA: [UNK] morton 
Human:  okay good bye
KATTA: hey 
Human:  what
KATTA: do not let me go 
Human:  goodbye
KATTA: alright 
Human:  END CONVO
KATTA: threatening me 
```