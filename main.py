import os
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt

# Disable all TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint


# File paths
data_dir = "gutenberg_books"
checkpoint_path = "gru_text_model.keras"
mapping_path = "char_mappings.pkl"

# --- Text generation function ---
def generate_text(model, seed_text, char_to_int, int_to_char, seq_length, length=200, temperature=1.0):
    vocab_size = len(char_to_int)
    generated = seed_text
    for _ in range(length):
        input_text = generated[-seq_length:]
        input_seq = np.zeros((1, seq_length), dtype=np.int32)
        for t, char in enumerate(input_text.rjust(seq_length)):
            input_seq[0, t] = char_to_int.get(char, 0)

        preds = model.predict(input_seq, verbose=0)[0]
        if preds.shape[0] != vocab_size:
            raise ValueError(f"Expected prediction size {vocab_size}, but got {preds.shape[0]}")

        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        next_index = np.random.choice(range(vocab_size), p=preds)
        next_char = int_to_char[next_index]
        generated += next_char

    return generated

# --- Main interactive function ---
def main():
    seq_length = 40
    embedding_dim = 50
    gru_units = 128

    if os.path.exists(checkpoint_path) and os.path.exists(mapping_path):
        choice = input("Model checkpoint found. Do you want to (T)rain or (G)enerate text? [T/G]: ").strip().upper()
    else:
        print("No saved model/mapping found. You must train first.")
        choice = "T"

    if choice == "G":
        print("Loading saved model and mappings...")
        model = load_model(checkpoint_path)
        with open(mapping_path, "rb") as f:
            char_to_int, int_to_char = pickle.load(f)

        vocab_size = len(char_to_int)

        seed = input("Enter seed text: ").strip()
        length = input("Enter number of characters to generate (default 200): ").strip()
        temperature = input("Enter temperature (default 1.0): ").strip()

        length = int(length) if length.isdigit() else 200
        try:
            temperature = float(temperature)
        except ValueError:
            temperature = 1.0

        print("\nGenerating text...\n")
        generated = generate_text(model, seed_text=seed.lower(), char_to_int=char_to_int, int_to_char=int_to_char,
                                  seq_length=seq_length, length=length, temperature=temperature)
        print(generated)

    else:
        print("Processing dataset...")
        all_text = ""
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    all_text += f.read().lower() + "\n"

        chars = sorted(list(set(all_text)))
        char_to_int = {c: i for i, c in enumerate(chars)}
        int_to_char = {i: c for i, c in enumerate(chars)}
        vocab_size = len(chars)

        # Save mappings
        with open(mapping_path, "wb") as f:
            pickle.dump((char_to_int, int_to_char), f)

        print(f"Total combined text length: {len(all_text)} characters")
        print(f"Vocabulary size: {vocab_size}")

        step = 3
        sequences = []
        next_chars = []
        for i in range(0, len(all_text) - seq_length, step):
            sequences.append(all_text[i:i + seq_length])
            next_chars.append(all_text[i + seq_length])

        print(f"Number of sequences: {len(sequences)}")

        X = np.zeros((len(sequences), seq_length), dtype=np.int32)
        y = np.zeros(len(sequences), dtype=np.int32)
        for i, seq in enumerate(sequences):
            X[i] = [char_to_int[ch] for ch in seq]
            y[i] = char_to_int[next_chars[i]]

        # Build and compile model
        if os.path.exists(checkpoint_path):
            print(f"Loading model from {checkpoint_path} ...")
            model = load_model(checkpoint_path)
        else:
            print("Building new model ...")
            model = Sequential([
                Embedding(vocab_size, embedding_dim, input_length=seq_length),
                GRU(gru_units),
                Dense(vocab_size, activation='softmax')
            ])
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train
        checkpoint_cb = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='loss',
            save_best_only=True,
            verbose=1
        )

        print("Starting training...")
        history = model.fit(X, y, batch_size=128, epochs=15, callbacks=[checkpoint_cb])

        # Plot training loss and accuracy
        plt.figure(figsize=(10, 4))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Loss')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Accuracy', color='orange')
        plt.title('Training Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig("training_plot.png")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()

