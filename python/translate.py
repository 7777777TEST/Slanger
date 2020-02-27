from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np
import json

latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 2560  # Number of samples to train on.
data_path = 'copas.txt'

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
input_characters.add(" ")
target_characters.add(" ")
with open(data_path, 'r', encoding='utf-8') as f:
	lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
	input_text,target_text = line.split('\t')
	target_text = '\t' + target_text + '\n'
	input_texts.append(input_text)
	target_texts.append(target_text)
	for char in input_text:
		if char not in input_characters:
			input_characters.add(char)
	for char in target_text:
		if char not in target_characters:
			target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
	[(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
	[(char, i) for i, char in enumerate(target_characters)])

# Save the token indices to file.
with open("metadada.json", 'w', encoding='utf-8') as f:
	metadata = {
		'input_token_index': input_token_index,
		'target_token_index': target_token_index,
		'max_encoder_seq_length': max_encoder_seq_length,
		'max_decoder_seq_length': max_decoder_seq_length
	}
	f.write(json.dumps(metadata, ensure_ascii=False))
print('Saved metadata at: metadata.json')

encoder_input_data = np.zeros(
	(len(input_texts), max_encoder_seq_length, num_encoder_tokens),
	dtype='float32')
decoder_input_data = np.zeros(
	(len(input_texts), max_decoder_seq_length, num_decoder_tokens),
	dtype='float32')
decoder_target_data = np.zeros(
	(len(input_texts), max_decoder_seq_length, num_decoder_tokens),
	dtype='float32')

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
									 initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
			  metrics=['accuracy'])
try:
	model.load_weights("model.h5")
	print("Load from model")
except:
	print("Error:Failed Loading from model")
	pass

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
	decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
	[decoder_inputs] + decoder_states_inputs,
	[decoder_outputs] + decoder_states)

reverse_input_char_index = dict(
	(i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
	(i, char) for char, i in target_token_index.items())

def decode_str(string):
	input_seq=np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
	for t, char in enumerate(string):
		input_seq[0, t, input_token_index[char]] = 1.
	input_seq[0, t+1:, input_token_index[' ']] = 1.

	states_value = encoder_model.predict(input_seq[0:1])
	target_seq = np.zeros((1, 1, num_decoder_tokens))
	target_seq[0, 0, target_token_index['\t']] = 1.

	stop_condition = False
	decoded_sentence = ''
	while not stop_condition:
		output_tokens, h, c = decoder_model.predict(
			[target_seq] + states_value)

		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]
		decoded_sentence += sampled_char

		if (sampled_char == '\n' ):
			stop_condition = True
		elif(len(decoded_sentence)>100):
			stop_condition=True

		target_seq = np.zeros((1, 1, num_decoder_tokens))
		target_seq[0, 0, sampled_token_index] = 1.

		# Update states
		states_value = [h, c]

	return decoded_sentence

while True:
	print("Output: "+decode_str(input("Input:")))