import keras
import numpy as np
import json
from tinysegmenter import TinySegmenter

data_path = 'copus.txt'
num_samples=100
num_epochs=100
batch_size=256

tokenize = TinySegmenter().tokenize
def preprocess_sentence(w):
	w=w.lower()
	tokens=tokenize(w)
	w="<start> "
	for word in tokens:
		if word==" ":
			continue
		w+=word+" "
	w+="<end>"
	return w

input_texts = []
target_texts = []
input_words = set()
target_words = set()
with open(data_path, 'r', encoding='utf-8') as f:
	lines = f.read().split('\n')
for line in lines:
	input_text,target_text = line.split('\t')
	target_text=preprocess_sentence(target_text)
	input_text=preprocess_sentence(input_text)
	input_texts.append(input_text)
	target_texts.append(target_text)
	for word in input_text.split(" "):
		if word not in input_words:
			input_words.add(word)
	for word in target_text.split(" "):
		if word not in target_words:
			target_words.add(word)

input_words = sorted(list(input_words))
target_words = sorted(list(target_words))
num_encoder_words = len(input_words)
num_decoder_words = len(target_words)
max_encoder_seq_length = max([len(txt.split(" ")) for txt in input_texts])
max_decoder_seq_length = max([len(txt.split(" ")) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input words:', num_encoder_words)
print('Number of unique output words:', num_decoder_words)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_word_index = dict([(word, i) for i, word in enumerate(input_words)])
target_word_index = dict([(word, i) for i, word in enumerate(target_words)])

# Save the token indices to file.
with open("metadata.json", 'w', encoding='utf-8') as f:
	metadata = {
		'input_word_index': input_word_index,
		'target_word_index': target_word_index,
		'max_encoder_seq_length': max_encoder_seq_length,
		'max_decoder_seq_length': max_decoder_seq_length
	}
	f.write(json.dumps(metadata, ensure_ascii=False))

print('Saved metadata at: metadata.json')

input_texts = []
target_texts = []

for line in lines[: min(num_samples, len(lines) - 1)]:
	input_text,target_text = line.split('\t')
	target_text=preprocess_sentence(target_text)
	input_text=preprocess_sentence(input_text)
	input_texts.append(input_text)
	target_texts.append(target_text)


def transform(encoding,data,vec_size):
	transformed_data=np.zeros((len(data),vec_size),dtype='int')
	for i in range(len(data)):
		for j in range(min(vec_size,len(data[i].split(" ")))):
			transformed_data[i][j]=encoding[data[i].split(" ")[j]]
	return transformed_data

encoded_training_input=transform(input_word_index,input_texts,max_encoder_seq_length)
encoded_training_output=transform(target_word_index,target_texts,max_decoder_seq_length)
training_encoder_input = encoded_training_input
training_decoder_input = np.zeros_like(encoded_training_output)
training_decoder_input[:, 1:] = encoded_training_output[:,:-1]
training_decoder_output = np.eye(num_decoder_words)[encoded_training_output.astype('int')]

def createModel(input_vocab_size,output_vocab_size,input_len,output_len):
	embedding_dim=128
	lstm_units=256
	encoder_input=keras.Input(shape=(input_len,))
	decoder_input=keras.Input(shape=(output_len,))
	encoder=keras.layers.Embedding(input_vocab_size, embedding_dim,input_length=input_len,mask_zero=True)(encoder_input)
	encoder=keras.layers.LSTM(lstm_units,return_sequences=True)(encoder)
	# encoder_last=encoder[:,-1,:]
	
	decoder=keras.layers.Embedding(output_vocab_size, embedding_dim,input_length=output_len,mask_zero=True)(decoder_input)
	decoder=keras.layers.LSTM(lstm_units,return_sequences=True)(decoder)#,initial_state=[encoder_last,encoder_last])
	attention=keras.layers.Dot(axes=[2,2])([decoder,encoder])
	attention=keras.layers.Activation(activation="softmax")(attention)
	context=keras.layers.Dot(axes=[2,1])([attention,encoder])
	decoder_combined_context=keras.layers.concatenate([context,decoder])
	output=keras.layers.TimeDistributed(keras.layers.Dense(lstm_units,activation="tanh"))(decoder_combined_context)
	output=keras.layers.TimeDistributed(keras.layers.Dense(output_vocab_size,activation="softmax"))(output)
	model=keras.Model([encoder_input,decoder_input],output)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()
	return model

model=createModel(num_encoder_words,num_decoder_words,max_encoder_seq_length,max_decoder_seq_length)
try:
	model.load_weights("model.h5")
	print("Loaded from model")
except:
	print("Error:Cannot load from model file")

model.fit(x=[training_encoder_input,training_decoder_input],y=[training_decoder_output],epochs=num_epochs,batch_size=batch_size,shuffle=True,validation_split=0.0)
model.save("model.h5")

reverse_input_word_index = dict(
	(i, word) for word, i in input_word_index.items())
reverse_target_word_index = dict(
	(i, word) for word, i in target_word_index.items())

def generate(text):
	text=preprocess_sentence(text)
	print(text)
	encoder_input = transform(input_word_index, [text], max_encoder_seq_length)
	print(encoder_input)
	decoder_input = np.zeros(shape=(len(encoder_input), max_decoder_seq_length))
	decoder_input[:,0] = target_word_index["<start>"]
	for i in range(1, max_decoder_seq_length):
		output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
		decoder_input[:,i] = (output[:,i])
	return decoder_input[:,1:]

def decode(sequence):
	text = ''
	print(sequence)
	for i in sequence[0]:
		text += reverse_target_word_index[i]+" "
		if reverse_target_word_index[i] == "<end>":
			break
	return text
while True:
	print("Output:",decode(generate(input("Input:"))))