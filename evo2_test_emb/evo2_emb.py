import torch # pyright: ignore[reportMissingImports]
from evo2 import Evo2 # pyright: ignore[reportMissingImports]
import os
import pandas as pd # pyright: ignore[reportMissingImports]
import Bio.SeqIO as SeqIO # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]
import argparse
from torch.utils.data import Dataset  # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]


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


def generate_embedding(model, sequence, layer_names:str, sliding_window:int):
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
        # Output embedding shape: [1, seq_len, 4096]
        output, embedding = model(tokenized_sequence, return_embeddings=True, layer_names=[layer_names])
    
    # return tensor with shape [4096] because we mean over first axis (sequence length) and squeeze batch axis
    return torch.mean(embedding[layer_names], dim=1).squeeze(0).detach().cpu()

def get_embedding(model, sequence, layer_names, sliding_window):
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
    embeddings = []
    sequence_length = len(sequence)
    if sequence_length < sliding_window:
        print('Sequence length smaller than sliding window -> provide to the model the complete genome sequence!\n')
        embeddings = generate_embedding(model, sequence, layer_names, sliding_window)
        return [embeddings, embeddings]

    # Use sliding window approach
    for i in range(0, sequence_length, sliding_window):
        # window sequence contains the subsequence for which we are going to generate the
        # embedding. It has max value if remaning genome sequence is longer than maximum
        # possible window, otherwise process remaining sequence.
        window_sequence = sequence[i:min(i+sliding_window, sequence_length)]
        # generate_embedding return an embedding with shape: [4096]
        embeddings.append(generate_embedding(model, window_sequence, layer_names, sliding_window))

    stack = torch.stack(embeddings)
    # return torch.sum(torch.stack(embeddings), dim=0)
    print(f'Shape of embeddings stack: {stack.shape}')
    return [torch.sum(stack, dim=0), torch.cat(embeddings, dim=0)]


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
    model = Evo2('evo2_7b')

    # Retrieve data from fasta files
    dataset = SequenceDataset(args.fasta_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    
    all_ids = []
    all_embeddings_sum = []
    all_embeddings_cat = []

    # Test dataset and dataloader
    print(f"Total sequences found: {len(dataset)}")
    #for index, pair in tqdm(enumerate(dataloader), desc="Computing Embeddings..."):
    for index, pair in enumerate(dataloader):
        id, sequence = pair

        # Resulting shape = [4096]
        embedding = get_embedding(model, str(sequence[0]), args.layer_names, args.window_size)
        all_ids.append(id[0])
        # Convert each element of the tensor to float32 for numpy compatibility
        all_embeddings_sum.append(embedding[0].float())
        all_embeddings_cat.append(embedding[1].float())

        print(f'Shape of summed embedding: {all_embeddings_sum[index].shape}')
        print(f'Shape of concatenated embedding: {all_embeddings_cat[index].shape}\n')
        
        if index == 5:
            # np.save(os.path.join(args.output_dir, 'test_embeddings_sum.npy'), all_embeddings_sum)
            # np.save(os.path.join(args.output_dir, 'test_embeddings_cat.npy'), all_embeddings_cat)
            break

    # 2. Stack and Convert ONCE at the end
    # Resulting shape: [N_sequences, 4096]
    #embedding_matrix = torch.stack(all_embeddings_cat).numpy() 
    embedding_matrix_cat = np.array(all_embeddings_cat, dtype=object)
    embedding_matrix_sum = np.array(all_embeddings_sum)
    id_array = np.array(all_ids)
    

    # 3. Save as .npz (Compressed)
    output_path_cat = os.path.join(args.output_dir, 'embeddings_cat.npz')
    np.savez_compressed(output_path_cat, embeddings=embedding_matrix_cat, ids=id_array)
    print(f"Saved .npz to {output_path_cat}")

    output_path_sum = os.path.join(args.output_dir, 'embeddings_sum.npz')
    np.savez_compressed(output_path_sum, embeddings=embedding_matrix_cat, ids=id_array)
    print(f"Saved .npz to {output_path_sum}")

    # 4. LOAD
    data = np.load(output_path_sum, allow_pickle=True)
    loaded_embeds = data['embeddings']
    loaded_ids = data['ids']

    print(f"Matrix Shape: {loaded_embeds.shape}")
    print('-----------')
    print(f"First Embedding Vector: {loaded_embeds[0]}")
    print(f'First Embedding Vector Shape: {loaded_embeds[0].shape}')
    print(f"First ID: {loaded_ids[0]}")
    print('-----------')
    
    print(f"First Embedding Vector: {loaded_embeds[1]}")
    print(f'First Embedding Vector Shape: {loaded_embeds[1].shape}')
    print(f"First ID: {loaded_ids[1]}")
    print('-----------')

    print(f"First Embedding Vector: {loaded_embeds[5]}")
    print(f'First Embedding Vector Shape: {loaded_embeds[5].shape}')
    print(f"First ID: {loaded_ids[5]}")
    print('-----------')

