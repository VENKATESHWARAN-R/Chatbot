# %%
# Importing the required libraries for data preparation
import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# %%
def load_data(file_name: str, data_location: str = './data/'):
    """
    This function will be loading the data from the filenames which are given as input and return the list of lines from the data file
    input: file_name -> str, data_location -> str = ./data/ by default
    output: lines -> list data lines list from the input file
    """
    def fix_dir(dir_name: str):
        if dir_name[-1] == '/':
            return dir_name
        return dir_name + '/'
    
    data_file = fix_dir(data_location) + file_name
    with open(data_file, 'r', encoding='utf-8', errors='ignore') as dfile:
        lines = dfile.read().split('\n')
    
    print(f'Data read from {data_file} and converted into {len(lines)} lines')

    return lines

# %%
def prepare_data(movie_titles: list, movie_conversations: list, movie_lines: list):
    """
    This function prepares data dictionary for each files it outputs list of dictionaries for all the major datasets 
    inputs: movie_titles -> list, movie_conversations -> list, movie_lines -> list
    outputs: movie_title_list -> list(dict), movie_conversation_list -> list(dict), movie_lines_list -> list(dict)
    """
    # Prepare dictionary for movie meta data
    movie_title_list = []
    for line in movie_titles:
        if not line:
            continue # for identifying and ignoring empty lines
        movie_title_info = {}
        movie_info = line.split(' +++$+++ ')
        movie_title_info['movie_id'] = movie_info[0].strip()
        movie_title_info['name'] = movie_info[1].strip()
        movie_title_info['year'] = movie_info[2].strip()
        movie_title_info['rating'] = movie_info[3].strip()
        movie_title_info['genre'] = movie_info[-1][2:-2].strip().split("', '") # this is for splitting the genres from ['comedy', 'romance'] to a list
        movie_title_list.append(movie_title_info)

    # Prepare dictionary for movie convo meta data
    movie_conversation_list = []
    for line in movie_conversations:
        if not line:
            continue # for identifying and ignoring empty lines
        movie_conversation_info = {}
        conversation_info = line.split(' +++$+++ ')
        movie_conversation_info['speaker1'] = conversation_info[0].strip()
        movie_conversation_info['speaker2'] = conversation_info[1].strip()
        movie_conversation_info['movie_id'] = conversation_info[2].strip()
        movie_conversation_info['line_ids'] = conversation_info[-1][2:-2].strip().split("', '")# this is for splitting the conversation info from ['L198', 'L199'] to a list
        movie_conversation_list.append(movie_conversation_info)

    # Prepare dictionary for movie dialogues
    movie_lines_list = []
    for line in movie_lines:
        if not line:
            continue # for identifying and ignoring empty lines
        movie_line_info = {}
        line_info = line.split(' +++$+++ ')
        movie_line_info['line_id'] = line_info[0].strip()
        movie_line_info['speaker'] = line_info[1].strip()
        movie_line_info['movie_id'] = line_info[2].strip()
        movie_line_info['character'] = line_info[3].strip()
        movie_line_info['dialogue'] = line_info[-1].strip()
        movie_lines_list.append(movie_line_info)

    return movie_title_list, movie_conversation_list, movie_lines_list

# %%
def dataframe_from_dict(data_dict_list: list):
    """
    This function converts the list of dictionaries into pandas dataframe
    input: data_dict_list -> list(dict)
    output: pandas dataframe prepared from the list
    """
    return pd.DataFrame.from_dict(data_dict_list)

# %%
def get_genre_dict(movie_title_df: pd.DataFrame):
    """
    This line takes the input as movie titles pandas dataframe and prepares the genre dict
    input: movie_title_df -> pandas.DataFrame
    output: genre_dict -> dict the output will have the dictionary with keys as genre and values as list of movies from that genre
    """
    # Get the list of available genres from the whole dataset 
    genres = movie_title_df['genre'].to_numpy()
    genre_set = set()
    for genre_list in genres:
        for genre in genre_list:
            if genre:
                genre_set.add(genre)
    
    # Checking the count of movies in each genres and storing the movies with respect to their genres in the dictionary
    genre_dict = {}
    for genre_name in genre_set:
        genre_dict[genre_name] = []
    for movie, genre_list in movie_title_df[['movie_id', 'genre']].to_numpy():
        for genre in genre_list:
            if genre:
              genre_dict[genre].append(movie)
    
    print('Genre dictionary prepared')

    return genre_dict

# %%
def prepare_conversations(movie_lines_df: pd.DataFrame, movie_conversation_df: pd.DataFrame, only_start: bool = False):
    """
    This line takes the input as movie lines pandas dataframe and prepares the genre dict
    input: movie_lines_df -> pandas.DataFrame, movie_conversation_df -> pandas.DataFrame
    output: dialogue_dict -> dict dictionary with line_id as key and respective line as value, conversation_data_df -> pandas.DataFrame will have question and answers dataframe
    """
    # Make conversation line dictionary for preparing the final dataset
    dialogue_ids = movie_lines_df['line_id'].to_numpy()
    dialogue_lines = movie_lines_df['dialogue'].to_numpy()
    dialogue_dict = {}
    for dialogue_id, dialogue_line in zip(dialogue_ids, dialogue_lines):
        dialogue_dict[dialogue_id] = dialogue_line

    # prepare final/actual dictionary for creating the chat bot
    # This dictionary will have the conversation wise data.
    conversation_data_dict = {}
    conversation_data_dict['movie_id'] = []
    conversation_data_dict['input'] = []
    conversation_data_dict['target'] = []
    for movie_id, convo_list in movie_conversation_df[['movie_id', 'line_ids']].to_numpy():
        for convos in range(len(convo_list)-1):
            conversation_data_dict['movie_id'].append(movie_id)
            conversation_data_dict['input'].append(dialogue_dict[convo_list[convos]])
            conversation_data_dict['target'].append(dialogue_dict[convo_list[convos+1]])
            if only_start:
              break

    # Prepare dataframe from the dictionary for better access
    conversation_data_df = pd.DataFrame.from_dict(conversation_data_dict)
    print('Conversations prepared')
    
    return dialogue_dict, conversation_data_df

# %%
# create a function for data cleaning
def clean_text(input_text: str, add_tags: bool = False, start_tag: str = 'START_ ', end_tag: str = ' _END', 
                remove_punc: bool = True, remove_symbols: str = '[^0-9a-z #+_]', ignore_words: list = [], 
                remove_numbers: bool = True, replace_word_from: list = [], replace_word_to: list = []):
    """
    Input: input_text (string), add_tags (optional - bool), start_tag (optional - string), end_tag (optional - string), 
            remove_punc (optional - bool), remove_symbols (optional - string), ignore_words (optional - list), remove_numbers (optional - bool),
            replace_word_from (optional - bool), replace_word_to (optional - bool)
    Output: cleaned text (string)
    description:
        This function will clean the input text given by removong the bad symbols, numbers, punctuations, extra spaces... and return back the cleaned text
        if the add_tags value is True (it's False by default) it will add the start tag and end tags at the start and end of the text
        we can also define the start_tag and end_tag values
    """
    def replace_common_words(text: str):
        text = text.lower()
        text = re.sub("i'm", "i am", text)
        text = re.sub("he's", "he is", text)
        text = re.sub("she's", "she is", text)
        text = re.sub("that's", "that is", text)
        text = re.sub("what's", "what is", text)
        text = re.sub("where's", "where is", text)
        text = re.sub("'ll", " will", text)
        text = re.sub("'ve", " have", text)
        text = re.sub("'re", " are", text)
        text = re.sub("'d", " would", text)
        text = re.sub("n't", " not", text)
        return text

    def remove_punctuation(text: str):
        punctuation_list = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in punctuation_list)

    def remove_bad_symbols(text: str, symbols: str):
        bad_symbols = re.compile(symbols)
        return bad_symbols.sub(' ', text)

    def remove_extra_space(text: str):
        extra_space = re.compile(' +')
        return extra_space.sub(' ', text)

    def remove_ignore_words(text: str, ignore_words_list: list):
        for word in ignore_words_list:
            text = text.replace(word, " ")
        return text
    
    def remove_digits(text:str):
        remove_digit = str.maketrans('', '', string.digits)
        return text.translate(remove_digit)

    def replace_words(text: str, replace_word_list_from: list, replace_word_list_to: list):
        for from_word, to_word in zip(replace_word_list_from, replace_word_list_to):
            text = text.replace(str(from_word).lower(), str(to_word).lower())
        return text

    def add_start_end_tags(text: str):
        return start_tag + text + end_tag

    input_text = input_text.lower()
    input_text = replace_common_words(input_text)
    input_text = replace_words(input_text, replace_word_from, replace_word_to) if replace_word_from and (len(replace_word_from) == len(replace_word_to)) else input_text
    input_text = remove_ignore_words(input_text, ignore_words) if ignore_words else input_text
    input_text = remove_digits(input_text) if remove_numbers else input_text
    input_text = remove_punctuation(input_text) if remove_punc else input_text
    input_text = remove_bad_symbols(input_text, remove_symbols) if remove_symbols else input_text
    input_text = add_start_end_tags(input_text) if add_tags else input_text
    input_text = remove_extra_space(input_text)
    #print('Data cleaning done')
    
    return input_text.strip()

# %%
def filter_short_long(conversation_data_df: pd.DataFrame, min_q_length: int = 2, max_q_length: int = 25, min_a_length: int = 2, max_a_length: int = 25):
    """
    This function takes list of input dialogues and list of target dialogues and returns only the dialogues with given length
    input: conversation_data_df -> pandas.DataFrame
    output: filtered_conversation_df -> pandas.DataFrame
    """
    movie_id_seq, qseq, aseq = conversation_data_df['movie_id'].to_numpy(), conversation_data_df['input'].to_numpy(), conversation_data_df['target'].to_numpy()
    conversation_data_dict = {}
    conversation_data_dict['movie_id'], conversation_data_dict['input'], conversation_data_dict['target'] = [], [], []
    raw_data_len = len(movie_id_seq)

    for i in range(raw_data_len):
        qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= min_q_length and qlen <= max_q_length:
            if alen >= min_a_length and alen <= max_a_length:
                conversation_data_dict['movie_id'].append(movie_id_seq[i])
                conversation_data_dict['input'].append(qseq[i])
                conversation_data_dict['target'].append(aseq[i])
    
    filt_data_len = len(conversation_data_dict['movie_id'])
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(f'{filtered}% filtered from original data')

    return pd.DataFrame.from_dict(conversation_data_dict)

# %%
def split_vectorize_filter_unk(conversation_data_df: pd.DataFrame, Vectorizer: TextVectorization, unk: str = '[UNK]', test_split: float = 0.2, seed: int = 42):
    """
    This function takes list of input dialogues and list of target dialogues and returns only the dialogues with less unknown tokens
    input: conversation_data_df -> pandas.DataFrame, vectorizer object
    output: training_data -> dict data needed for training, testing_data -> data needed for testing
    """
    def remove_start_tag(input_with_start_tag: str):
        return ' '.join(input_with_start_tag.split()[1:])

    movie_id_seq, qseq, aseq = conversation_data_df['movie_id'].to_numpy(), conversation_data_df['input'].to_numpy(), conversation_data_df['target'].to_numpy()
    training_data = {}
    testing_data = {}
    training_data['input'], training_data['target'], training_data['input_vectors'], training_data['target_vectors'] = [], [], [], []
    testing_data['input'], testing_data['target'], testing_data['input_vectors'], testing_data['target_vectors'] = [], [], [], []

    raw_data_len = len(movie_id_seq)
    vocab_list = Vectorizer.get_vocabulary()
    unk_index = vocab_list.index(unk)

    train_inputs, test_inputs, train_targets, test_targets = train_test_split(qseq, aseq, test_size=test_split, random_state=seed)
    
    start_tag_removed_train_targets = [remove_start_tag(target) for target in train_targets]
    start_tag_removed_test_targets = [remove_start_tag(target) for target in test_targets]

    train_vectorized_inputs, train_vectorized_targets = Vectorizer(train_inputs), Vectorizer(start_tag_removed_train_targets)
    test_vectorized_inputs, test_vectorized_targets = Vectorizer(test_inputs), Vectorizer(start_tag_removed_test_targets)

    for idx, (input_tensor, target_tensor) in enumerate(zip(train_vectorized_inputs, train_vectorized_targets)):
        input_list = list(input_tensor.numpy())
        target_list = list(target_tensor.numpy())
        unknown_count_q = input_list.count(unk_index)
        unknown_count_a = target_list.count(unk_index)
        if unknown_count_a <=1 :
            if unknown_count_q > 0:
                temp_list = list(filter(lambda num: num != 0, input_list)) # This list will have the inputs without zeros padded
                if unknown_count_q/len(temp_list) > 0.2:
                    continue
            training_data['input'].append(train_inputs[idx])
            training_data['target'].append(train_targets[idx])
            training_data['input_vectors'].append(input_tensor)
            training_data['target_vectors'].append(target_tensor)
        
    testing_data['input'], testing_data['target'] = test_inputs, test_targets 
    testing_data['input_vectors'], testing_data['target_vectors'] = test_vectorized_inputs, test_vectorized_targets

    print(f'Training data points: {len(train_inputs)}')
    print(f'Test data points: {len(test_inputs)}')
    filt_data_len = len(training_data['input'])
    filtered = int((len(train_inputs) - filt_data_len)*100/len(train_inputs))
    print(f'{filtered}% filtered from training data points')
    print(f'After unknown token filters training data points: {filt_data_len}')

    return training_data, testing_data

# %%



