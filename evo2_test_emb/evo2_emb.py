import torch
from evo2 import Evo2
import os
import pandas as pd
import Bio.SeqIO as SeqIO
from torch.utils.data import DataLoader, Dataset
import argparse
import numpy as np
from datetime import datetime

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


def generate_embedding(model, sequence, layer_names):
    """
    Compute embedding for a sequence using Evo2 model.
    Args:
        model: Instance of Evo2 model.
        sequence (str): The input sequence.
        layer_names (str): The layer name to extract embeddings from.
        sliding_window (int): The size of the sliding window.
    Returns:
        torch.Tensor: The resulting embedding tensor. Shape: [4096]
    """
    # Tokenized sequence shape: [1, seq_len, 4096]
    tokenized_sequence = torch.tensor(
        model.tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to('cuda')

    with torch.inference_mode():
        _, embedding = model(tokenized_sequence, return_embeddings=True, layer_names=[layer_names])
    # Output embedding shape: [1, seq_len, 4096]
    # return tensor with shape [4096] because we mean over first axis (seq_len) and squeeze batch axis
    return torch.mean(embedding[layer_names], dim=1).squeeze(0).detach().cpu()

def get_embedding(model, sequence, layer_names, sliding_window, stride):
    """
    Return an embedding for a sequence using sliding window approach if necessary.
    It automatically manage sequences longer than sliding_window size.
    Args:
        model: Instance of Evo2 model.
        sequence (str): The input sequence.
        layer_names (str): The layer name to extract embeddings from.
        sliding_window (int): The size of the sliding window.
    Returns:
        torch.Tensor: The resulting embedding tensor. Shape: [4096]
    """
    sequence_length = len(sequence)
    if sequence_length < sliding_window:
        return generate_embedding(model, sequence, layer_names)

    # Use sliding window approach
    embeddings = []
    for start in range(0, sequence_length, stride):
        end = min(start + sliding_window, sequence_length)
        window_sequence = sequence[start : end]
        # generate_embedding return an embedding with shape: [4096]
        embeddings.append(generate_embedding(model, window_sequence, layer_names))

    return torch.sum(torch.stack(embeddings), dim=0)

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
    parser.add_argument("--stride", type=int, default=32768, dest="stride",
                        help="Value of the stride (default: 32768)")
    # Parse arguments
    args = parser.parse_args()
    batch_size = 1

    # Initialize the model and move it to the GPU
    model = Evo2('evo2_7b')

    # Retrieve data from fasta files
    dataset = SequenceDataset(args.fasta_dir)
    # Test dataset and dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_ids = []
    all_embeddings = []
    
    date = datetime.now()
    # String format: date_time_dataset-length_window-size_layer-name_stride
    output_path = os.path.join(args.output_dir, 
                               f'embeddings_{date.year}-{date.month}-{date.day}_{date.hour}-{date.minute}-{date.second}_{len(dataset)}_{args.window_size}_{args.layer_names}_{args.stride}.pt')
    
    try:
        for index, pair in enumerate(dataloader):
            
            # We need to clear cache every n iterations, otherwise. 
            if index % 100 == 0:
                torch.cuda.empty_cache()

            id, sequence = pair

            # Resulting shape = [4096]
            embedding = get_embedding(model, str(sequence[0]), args.layer_names, args.window_size, args.stride)
            # We took item at 0 because id is a list. 
            # Our fasta files contains only one genome each, so the lists will containonly one element
            all_ids.append(id[0])
            all_embeddings.append(embedding.float())

            # Save progress every 50 iterations
            if index % 100 == 0:
                torch.save(
                    {
                        "ids": all_ids,
                        "embeddings": all_embeddings,
                    },
                    output_path
                )

    except Exception as e:
        # Save right before crashing
        torch.save(
            {
                "ids": all_ids,
                "embeddings": all_embeddings,
            },
            output_path
        )
        print(f"Crash at index {len(all_ids)}, progress saved to {output_path}")
        raise e

    # Sanity check
    assert(len(all_ids) == len(all_embeddings)), "Error: length mismatch between IDs and Embeddings counts"

    # Stack and convert to numpy array, together with IDs
    # Resulting shape: [N_sequences, 4096]
    embedding_matrix = torch.stack(all_embeddings).numpy() 
    id_array = np.array(all_ids)

    torch.save(
        {
            "ids": all_ids,
            "embeddings": all_embeddings,
        },
        output_path
    )