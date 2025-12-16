import torch
from evo2 import Evo2
import os
import pandas as pd
import Bio.SeqIO as SeqIO
from torch.utils.data import DataLoader
import argparse
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, fasta_dir):
        self.index = {}
        for fname in os.listdir(fasta_dir):
            if fname.endswith((".fa", ".fasta")):
                path = os.path.join(fasta_dir, fname)
                self.index.update(SeqIO.index(path, "fasta"))

        self.keys = list(self.index.keys())
                        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        record = self.index[key]
        sequence = str(record.seq)
        return record.id, sequence


def generate_embedding(evo2_model, sequence: str, layer_names:str, sliding_window:int):
    tokenized_sequence = torch.tensor(
        evo2_model.tokenizer.tokenize(sequence[:10000]),
        dtype=torch.int,
    ).unsqueeze(0).to('cuda')
    with torch.inference_mode():
        output, embedding = evo2_model(tokenized_sequence, return_embeddings=True, layer_names=[layer_names])
    return embedding



if __name__ == "__main__":

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Command line parser for Evo 2.")
    parser.add_argument("--data", dest="fasta_dir",
                        help="Input directory containing FASTA files", metavar="DIR")
    parser.add_argument("--output", dest="output_dir",
                        help="Output directory to save embeddings", metavar="DIR")
    parser.add_argument("--window_size", type=int, default=10000, dest="window_size",
                        help="Context window size (default: 10000)")
    parser.add_argument("--model_name", type=str, default="evo2_7b", dest="model_name",
                        help="Evo2 model name (default: evo2_7b)")
    parser.add_argument("--layer_name", type=str, default="blocks.28.mlp.l3", dest="layer_names",
                        help="Layer name to extract embeddings from (default: blocks.28.mlp.l3)")
    # Parse arguments
    args = parser.parse_args()
    batch_size = 1

    # Initialize the model and move it to the GPU
    evo2_model = Evo2('evo2_7b')

    # Retrieve data from fasta files
    dataset = SequenceDataset(args.fasta_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Test dataset and dataloader
    print(f"Total sequences found: {len(dataset)}")
    for index, pair in enumerate(dataloader):
        id, sequence = pair
        
        print(f"Index: {index}")
        print(f"ID: {id}, Sequence Length: {len(sequence[0])}")
        print(f"Sequence (first 50 bases): {sequence[0]}")
        print(f"Type of sequence: {type(sequence[0])}")
        #sequence_str = str(sequence[0])

        embedding = generate_embedding(evo2_model, str(sequence[0]), args.layer_names, args.window_size)
        print('Embeddings shape: ', embedding[args.layer_names].shape)
        print('\nEmbeddings:', embedding)
        if index == 10:
            break
        # # Move sequences to GPU and get embeddings
        # tokenized_sequence = torch.tensor(
        #     evo2_model.tokenizer.tokenize(sequence_str[:10000]),
        #     dtype=torch.int,
        # ).unsqueeze(0).to('cuda')
        # with torch.inference_mode():
        #     output, embedding = evo2_model(tokenized_sequence, return_embeddings=True, layer_names=[args.layer_names])

        #     print('Embeddings shape: ', embedding[args.layer_names].shape)
        #     print('Output', output)
        #     print('\nEmbeddings:', embedding)  
        #     print('\nEmbeddings tensor:', embedding[args.layer_names])
        #     break # Remove this break to process all sequences 