import numpy as np
import csv

from sklearn.metrics.pairwise import cosine_similarity
import keras
from keras import layers
from keras.optimizers import Adam

from collections import Counter

#hyperparameters 
learning_rate = 0.01
epochs = 150
batch_size = 8
emb_size = 10
context_size = 4


def write_data(data):
    with open('cbow_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        
        # Write all rows at once
        writer.writerow(['inputs', 'outputs'])
        for row in zip(data[0], data[1]):
            writer.writerow([row[0], row[1]])

def write_vocab(data):
    with open('vocab.txt', 'w') as file:
        occurrences = Counter(data)
        for word, count in occurrences.items():
            file.write(f"{word}: {count}\n")


with open("input.txt") as f:
    data = f.read()

punct = [',', '\n', '.']
clean_data = ""

for ch in data:
    if ch not in punct:
        clean_data += ch

clean_data = clean_data.split()
write_vocab(clean_data)


vocab = list(set(clean_data))
enc = {word: idx for idx, word in enumerate(vocab+['<pad>'])}
dec = {idx: word for idx, word in enumerate(vocab+['<pad>'])}
vocab_size = len(vocab) + 1  # `+ 1` for <pad>

x_dec = []
y_dec = []

for i in range(len(clean_data)):
    start = max(0, i - context_size)
    end = min(len(clean_data), i + (context_size+1))

    context = clean_data[start:i]
    context.extend(clean_data[i+1:end])

    word = [clean_data[i]]

    x_dec.append(context)
    y_dec.append(word)


x_enc = []
y_enc = []

for inp,out in zip(x_dec,y_dec):
    x_enc.append([enc[val] for val in inp])
    y_enc.append(enc[out[0]])

for i in range(len(x_enc)):
    if len(x_enc[i]) < 8:
        x_enc[i] = x_enc[i] + [enc['<pad>']]*(8-len(x_enc[i]))

write_data([x_enc, y_enc])

x_enc = np.array(x_enc)
y_enc = np.array(y_enc)


model_cbow = keras.Sequential([
    # This is the layer that generates the embedding vectors!
    layers.Embedding(input_dim=vocab_size, output_dim=emb_size, name="word_embeddings"),
    
    # If using CBOW, you'd add GlobalAveragePooling1D() here
    layers.GlobalAveragePooling1D(), 
    
    # Output layer to predict the target word
    layers.Dense(vocab_size, activation="softmax")
])

model_cbow.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the Model
model_cbow.fit(
    x_enc, 
    y_enc, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_split=0.2, # Automatically uses 20% of data for validation
    verbose=1
)

# Extract the weights from the first layer
# shape will be (vocab_size, embed_dim)
embeddings = model_cbow.layers[0].get_weights()[0]

print("\n\n\n\n\n------------------------Testing Similarity------------------------")
find = input("Enter a word to find similar words: ")

# Find top 5 most similar words to 'physician'
# This compares vec_a against every row in the embedding_matrix
all_similarities = cosine_similarity(embeddings[enc[find]].reshape(1, -1), embeddings)[0]

# Get indices of the highest scores (excluding the word itself)
nearest_indices = all_similarities.argsort()[-6:-1][::-1]

print(f"\nTop 5 words similar to `{find}` using CBOW:\n")

# Map indices back to words
for i in nearest_indices:
    print(f"Word: `{dec[i]}`, Score: {all_similarities[i]}")