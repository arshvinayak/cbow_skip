import numpy as np
import csv
import argparse

from sklearn.metrics.pairwise import cosine_similarity
import keras
from keras import layers
from keras.optimizers import Adam

from collections import Counter

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Skip-gram model hyperparameters")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--num_negatives", type=int, default=10, help="Number of negative samples")
args = parser.parse_args()

#hyperparameters 
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
num_negatives = args.num_negatives

emb_size = 10
context_size = 4


def write_data(data):
    with open('skipgram_dataset.csv', 'w', newline='') as file:
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

def write_loss(history):
    with open('loss_skipgram.txt', 'w') as file:        
        # Write all rows at once
        file.write('epoch,loss,val_loss\n')
        for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            file.write(f"{epoch+1},{loss: .3f},{val_loss: .3f}\n")


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

#cbow
for i in range(len(clean_data)):
    start = max(0, i - context_size)
    end = min(len(clean_data), i + (context_size+1))

    context = clean_data[start:i]
    context.extend(clean_data[i+1:end])

    word = [clean_data[i]]

    x_dec.extend([[word[0], con] for con in context])
    
    y_dec.extend([1]*len(context))

#implementing negative sampling
x_dec_ = []
for word, con in x_dec:
  # 2. Add 'k' negative samples
  negatives_found = 0
  while negatives_found < num_negatives:
      # Randomly pick an item
      neg_con = np.random.choice(vocab)
      
      # Check if this is a "true" negative (user never interacted with it)
      if [word, str(neg_con)] not in x_dec:
        x_dec_.append([word, str(neg_con)])
        y_dec.append(0)
        negatives_found += 1

x_dec.extend(x_dec_)

x_enc = []
y_enc = y_dec

for inp,out in zip(x_dec,y_dec):
    x_enc.append([enc[val] for val in inp])

write_data([x_enc, y_enc])

x_enc = np.array(x_enc)
y_enc = np.array(y_enc)


# 1. Inputs
input_target = layers.Input((1,))
input_context = layers.Input((1,))

# 2. Shared Embedding Layer
# Both target and context words share the same lookup table
embedding = layers.Embedding(vocab_size, emb_size, name="word_embedding")

# Look up the vectors
target_vector = embedding(input_target)
context_vector = embedding(input_context)

# 3. Dot Product (Similarity Calculation)
# This calculates how 'aligned' the two vectors are
dot_product = layers.Dot(axes=2)([target_vector, context_vector])
flatten = layers.Flatten()(dot_product)

# 4. Output Layer (Sigmoid)
# Predicts 1 if they belong together, 0 if they don't
output = layers.Dense(1, activation="sigmoid")(flatten)

# 5. Build and Compile
model_skip = keras.Model(inputs=[input_target, input_context], 
                    outputs=output)

model_skip.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)

# train the Model
history = model_skip.fit([x_enc[:,0], x_enc[:,1]], 
          y_enc,
          epochs=epochs, 
          batch_size=batch_size, 
          validation_split=0.2, # Automatically uses 20% of data for validation
          verbose=1
)

write_loss(history)


# Extract the weights from the first layer
# shape will be (vocab_size, embed_dim)
embeddings = model_skip.layers[2].get_weights()[0]

print("\n\n\n\n\n------------------------Testing Similarity------------------------")
find = input("Enter a word to find similar words: ")

# Find top 5 most similar words to 'physician'
# This compares vec_a against every row in the embedding_matrix
all_similarities = cosine_similarity(embeddings[enc[find]].reshape(1, -1), embeddings)[0]

# Get indices of the highest scores (excluding the word itself)
nearest_indices = all_similarities.argsort()[-6:-1][::-1]

print(f"\nTop 5 words similar to `{find}` using Skip-Gram:\n")
# Map indices back to words
for i in nearest_indices:
    print(f"Word: `{dec[i]}`, Score: {all_similarities[i]}")