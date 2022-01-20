# %% [markdown]
# # Setting up the environment
# 
# > Loading the libraries & Dependencies

# %%
print('Loading the libraries')
print('...')
print('...')

# %%
# Importing the required notebooks
#import import_ipynb
import data_utils

# %%
# Importing the required libraires
import pickle
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# %%
print('Libraries loaded \n')

# %%
print('Setting up the variables and objects \n')

# %%
# Setting up the variable for preparing the model
random_seed = 42
embedding_output_dimension = 128
lstm_units = 400
stacked_lstm_units = 256
sparse_loss_fun = 'sparse_categorical_crossentropy'
one_hot_loss_fuc = 'categorical_crossentropy'

# %%
print('Preparing Vectorizer')
print('...')
print('...')

# %%
# Load the saved vectorizer
Vectorizer_dump = pickle.load(open('dependencies/vectorizer.pkl', 'rb'))
vectorizer_config = Vectorizer_dump['config']
max_vocab_length = vectorizer_config['max_tokens']
output_mode = vectorizer_config['output_mode']
max_length = vectorizer_config['output_sequence_length']
standardizer = vectorizer_config['standardize']
#Vectorizer = TextVectorization.from_config(Vectorizer_dump['config'])
# Prepare text vectorizer object
# Following need to be ran in order to prepare the model. The from_config method won't work while creating model as it's not defined in runtime
Vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                output_mode="int",
                                output_sequence_length=max_length,
                                standardize=None)
# Below line need to be included incase of older version of tensorflow
Vectorizer.adapt(['xyz'])
Vectorizer.set_weights(Vectorizer_dump['weights'])
vocab_list = Vectorizer_dump['vocab_list']

# %%
print('Vectorizer Prepared \n')

# %% [markdown]
# # Model generation

# %%
print('Generating the model')
print('...')
print('...')

# %%
# Creatimg embedding objects for encoder and decoder models
EncoderEmbeddingLayerM4 = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=embedding_output_dimension,
                                     input_length=max_length,
                                     mask_zero=True,
                                     name='encoder_embedding_layer_model4')

DecoderEmbeddingLayerM4 = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=embedding_output_dimension,
                                     input_length=max_length,
                                     mask_zero=True,
                                     name='decoder_embedding_layer_model4')

# %%
# create encoder & decoder initial layers
EncoderInputM4 = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
encoder_vectorM4 = Vectorizer(EncoderInputM4)

DecoderInputM4 = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
decoder_vectorM4 = Vectorizer(DecoderInputM4)

# %%
# create encoder
encoder_embeddings_M4 = EncoderEmbeddingLayerM4(encoder_vectorM4)

EncoderLstmLayer1M4 = tf.keras.layers.LSTM(stacked_lstm_units, return_state=True, return_sequences=True, name='Encoder_LSTM_layer1_model4')
encoder_outputs_layer1_M4, state_h_layer1_M4, state_c_layer1_M4 = EncoderLstmLayer1M4(encoder_embeddings_M4)
encoder_states_layer1_M4 = [state_h_layer1_M4, state_c_layer1_M4]

EncoderLstmLayer2M4 = tf.keras.layers.LSTM(stacked_lstm_units, return_state=True, name='Encoder_LSTM2_layer2_model4')
encoder_outputs_layer2_M4, state_h_layer2_M4, state_c_layer2_M4 = EncoderLstmLayer2M4(encoder_outputs_layer1_M4)
encoder_states_layer2_M4 = [state_h_layer2_M4, state_c_layer2_M4]

# %%
# create decoder
decoder_embeddings_M4 = DecoderEmbeddingLayerM4(decoder_vectorM4)

DecoderLstmLayer1M4 = tf.keras.layers.LSTM(stacked_lstm_units, return_sequences=True, return_state=True, name='Decoder_LSTM2_layer1_model4')
decoder_outputs_layer1_M4, _, _ = DecoderLstmLayer1M4(decoder_embeddings_M4, initial_state=encoder_states_layer1_M4)

DecoderLstmLayer2M4 = tf.keras.layers.LSTM(stacked_lstm_units, return_sequences=True, return_state=True, name='Decoder_LSTM2_layer2_model4')
decoder_outputs_layer2_M4, _, _ = DecoderLstmLayer2M4(decoder_outputs_layer1_M4, initial_state=encoder_states_layer2_M4)

DecoderDenseLayerM4 = tf.keras.layers.Dense(max_vocab_length, activation='softmax', name='Decoder_Dense_layer_model4')
decoder_dense_outputs_M4 = DecoderDenseLayerM4(decoder_outputs_layer2_M4)

EncDecModel4 = tf.keras.Model([EncoderInputM4, DecoderInputM4], decoder_dense_outputs_M4)

# %%
# Compile model
EncDecModel4.compile(loss=sparse_loss_fun,
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])

# %%
# Encoder at test time
EncModel4 = tf.keras.Model(EncoderInputM4, [encoder_states_layer1_M4, encoder_states_layer2_M4])

# %%
print('Model creation done! \n')

# %%
print('Loading the weights to the model')
print('...')
print('...')

# %%
# Loading model weights from best model
EncDecModel4.load_weights('model_weights/2LayerLstmAllData50epochs/EncDecModel4Weights')

# %%
print('Weights are loaded into the model suceessfully \n')

# %%
# Decoder function during runtime
def decoder_model4_test(input_word: str, states: list, vocab_list: list):
    decoder_vector_test_M4 = Vectorizer([input_word])
    dec_embedding_test_M4 = DecoderEmbeddingLayerM4(decoder_vector_test_M4)
    decoder_lstm_outputs_test_layer1_M4, state_h_l1_M4, state_c_l1_M4 = DecoderLstmLayer1M4(dec_embedding_test_M4, initial_state=states[0])
    decoder_lstm_outputs_test_layer2_M4, state_h_l2_M4, state_c_l2_M4 = DecoderLstmLayer2M4(decoder_lstm_outputs_test_layer1_M4, initial_state=states[1])
    decoder_dense_outputs_test_M4 = DecoderDenseLayerM4(decoder_lstm_outputs_test_layer2_M4)
    decoder_output = decoder_dense_outputs_test_M4[0, 0, :].numpy()
    word_idx = tf.argmax(decoder_output).numpy()
    next_word = vocab_list[word_idx]
    states[0] = [tf.constant(state_h_l1_M4), tf.constant(state_c_l1_M4)]
    states[1] = [tf.constant(state_h_l2_M4), tf.constant(state_c_l2_M4)]
    return decoder_output, next_word, states

# %%
print('Everything is ready')
print('You can start talking with KATTA')
print('For ending the conversation with KATTA use the keyword "END_CONVO"')

# %%
end_convo = True
while end_convo:
  human = input("Human: ")
  print('Human: ', human)
  if human == 'END_CONVO':
    end_convo = False
  bot_response = data_utils.make_prediction(vocab_list=vocab_list, decoder_model_function=decoder_model4_test, 
                                            encoder_model=EncModel4, input_text=human, clean_text=data_utils.clean_text, multi_layer=True,
                                            go_beam=True)
  print("KATTA:", bot_response)

# %%
"""
Human:  hi
KATTA: hello 
Human:  hi
KATTA: hello 
Human:  hi
KATTA: hello 
Human:  hi
KATTA: hello what do you want 
Human:  how are you
KATTA: all right 
Human:  how are you
KATTA: okay 
Human:  END CONVO
KATTA: threatening me 
"""


