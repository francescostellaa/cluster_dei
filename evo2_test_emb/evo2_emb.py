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


def generate_embedding(evo2_model, sequence, layer_names:str, sliding_window:int):
    # Tokenized sequence shape: [1, seq_len, 4096]
    tokenized_sequence = torch.tensor(
        evo2_model.tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to('cuda')
    with torch.inference_mode():
        output, embedding = evo2_model(tokenized_sequence, return_embeddings=True, layer_names=[layer_names])
    # Output embedding shape: [1, seq_len, 4096]
    # return tensor with shape [4096] because we mean over first axis (sequence length) and squeeze batch axis
    return torch.mean(embedding[layer_names], dim=1).squeeze(0).detach().cpu()

def get_embedding(evo2_model, sequence, layer_names, sliding_window):
    sequence_length = len(sequence)
    if sequence_length < sliding_window:
        return generate_embedding(evo2_model, sequence, layer_names, sliding_window)

    # Use sliding window approach
    embeddings = []
    for i in range(0, sequence_length, sliding_window):
        window_sequence = sequence[i:min(i+sliding_window, sequence_length)]
        # Embeddings shape: [4096]
        embedding = generate_embedding(evo2_model, window_sequence, layer_names, sliding_window)
        # print(f'Shape of return embedding: {embedding.shape}')
        embeddings.append(embedding)
    return torch.cat(embeddings)

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

    sequence_embeddings = {}

    # Test dataset and dataloader
    print(f"Total sequences found: {len(dataset)}")
    for index, pair in enumerate(dataloader):
        id, sequence = pair
        
        # print(f"Index: {index}")
        # print(f"ID: {id}, Sequence Length: {len(sequence[0])}")
        # print(f"Sequence (first 50 bases): {sequence[0]}")
        # print(f"Type of sequence: {type(sequence[0])}")
        #sequence_str = str(sequence[0])

        # tensor's size = [4096]
        embedding = get_embedding(evo2_model, str(sequence[0]), args.layer_names, args.window_size)
        sequence_embeddings.update({id : embedding})
        # print('\nEmbeddings:', embedding)
        if index == 10:
            torch.save(sequence_embeddings, os.path.join(args.output_dir, 'test_embeddings.pt'))
            break
        
    # Load embeddings to test if they were saved correctly
    loaded_embeddings = torch.load(os.path.join(args.output_dir, 'test_embeddings.pt'))
    print(f'Loaded embeddings keys: {list(loaded_embeddings.keys())}')