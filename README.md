# Word Prediction with GRU (Gated Recurrent Unit)

A lightweight and efficient word prediction model built using a GRU (Gated Recurrent Unit) neural network in PyTorch. This project demonstrates how sequence modeling can be applied to generate the next word(s) in a sentence using deep learning.

>  Designed for reproducibility, speed, and minimal setup using [`uv`](https://github.com/astral-sh/uv) for dependency management.

---

## Features

-  Sequence modeling with GRU layers in Tensorflow
-  Tokenization and vocabulary building
-  Training on a custom or provided text corpus
-  Predict next word(s) from a seed input
-  Fast setup with `uv`

---

## Installation

### Prerequisites

Make sure you have Python 3.10+ installed. Then, install [`uv`](https://github.com/astral-sh/uv):

After that install the dependencies using the command:
```bash
uv sync 
```

## Running

### Downloading text using Project Gutenberg library

Run the program gutenberg-downloader.py to download Gutenberg opensource books to the gutenberg_books folder for training. This program slowly downloads each english book but downloading the whole library as a zip file is possible in their website.

```bash
uv run gutenberg-downloader.py
```
You can also upload ur own text into the gutenberg_books folder

### Training/Running the model 

Run the main program using uv and follow the prompted messages for usage

```bash
uv run main.py
```
