import torch
from evo2 import Evo2
import os
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
        window_size (int): The size of the sliding window.
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

def get_embedding(model, sequence, layer_names, window_size, stride, summation, embedding_list: list, starts: list, ends: list):
    """
    Compute embeddings for the provided sequence and store them into provided list, with corresponding start and end indices in the sequence 
    It automatically manage sequences longer than window_size size.
    Args:
        model: Instance of Evo2 model.
        sequence (str): The input sequence.
        layer_names (str): The layer name to extract embeddings from.
        window_size (int): The size of the sliding window.
        stride (int): Stride
        summation (int): Tells if embeddings have to be summed to obtain only one of them or not
        emb (List[int]): list of embeddings
        starts (List[int]): list of start positions
        ends (List[int]): list of end positions
    """
    sequence_length = len(sequence)
    if sequence_length < window_size:
        starts.append(int(0))                   # Store first index
        ends.append(int(sequence_length))   # Store last index 
        embedding_list.append(generate_embedding(model, sequence, layer_names).float())
        return

    # Use sliding window approach
    embeddings = []
    for start in range(0, sequence_length, stride):
        end = min(start + window_size, sequence_length)
        window_sequence = sequence[start : end]
        starts.append(int(start))    # List containing all start positions for sequence's embeddings
        ends.append(int(end))    # List containing all end positions for sequence's embeddings
        # generate_embedding return an embedding with shape: [4096]
        embeddings.append(generate_embedding(model, window_sequence, layer_names))
    if summation:
        embedding_list.append(torch.sum(torch.stack(embeddings), dim=0).float())
    else: 
        embedding_list[:] = [item.float() for item in embeddings]
    return


def save_embeddings(all_ids, all_embeddings, output_path, summation, all_starts=None, all_ends=None):
    """
    Store data to specified output path.
    Args:
        all_ids (List): list of ids
        all_embeddigns (List): list of embeddings
        output_path (str): string that represent the path where ids and embeddings has to be saved 
        summation (int): Flag to know if the methods needs to save starts and ends
        all_stars (List): List of all start inidices for the embeddings
        all_ends (List): Llist of all end indices for the embeddings
    Returns:
        None
    Raise:
        Error if something goes wrong while saving operation
    """
    try:
        if int(summation) == int(1):
            torch.save(
                {
                    "ids": all_ids,
                    "embeddings": all_embeddings,
                },
                output_path
            )
        else:
            # Check for possible errors
            assert all_starts != None, "Cannot save starts list because it is None"
            assert all_ends != None, "Cannot save ends list because it is None"
            torch.save(
                {
                    "ids": all_ids,
                    "starts": all_starts,
                    "ends": all_ends,
                    "embeddings": all_embeddings,
                },
                output_path
            )
            
    except Exception as e:
        print("Error while saving data!")
        raise e
    return





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
    parser.add_argument("--summation", type=int, default=0, dest="summation",
                        help="Return the sum of context windows' embeddings if genome length is greater than context window size (default: 0(Not Active))")
    # Parse arguments
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    assert args.stride <= args.window_size, "stride must be <= window_size"
    batch_size = 1

    # Initialize the model and move it to the GPU
    model = Evo2(args.model_name)

    # Retrieve data from fasta files
    dataset = SequenceDataset(args.fasta_dir)
    # Test dataset and dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    all_ids = []
    all_embeddings = []
    # Following lists will be used only if args.summation == 1
    all_starts = [] # List of all start position for each embeddings
    all_ends = []   # List of all end position for each embeddings
        
        
    date = datetime.now()
    # String format: date_time_dataset-length_window-size_layer-name_stride
    output_path = os.path.join(
        args.output_dir, 
        f'embeddings_{date.year}-{date.month}-{date.day}_{date.hour}-{date.minute}-{date.second}_{len(dataset)}_{args.window_size}_{args.layer_names}_{args.stride}_{args.summation}.pt'
    )
    
    # TEST_LIMIT = 5
    
    try:
        for index, pair in enumerate(dataloader):
            # TEST: Stop loop if we exceed the limit
            # if index >= TEST_LIMIT:
            #     print(f"\n--- Test limit of {TEST_LIMIT} reached. Stopping loop. ---")
            #     break
            # We need to clear cache every n iterations, otherwise. 
            if index % 200 == 0:
                torch.cuda.empty_cache()

            id, sequence = pair
            # List of embeddings, starting and end positions for current sequence
            embeddings, starts, ends = [], [], []
            
            get_embedding(
                model,
                str(sequence[0]),
                args.layer_names,
                args.window_size,
                args.stride,
                args.summation,
                embeddings,
                starts,
                ends
            )
            
            assert len(embeddings) != 0, "Embeddings list is empty."
            assert len(starts) != 0, "Starts list is empty."
            assert len(ends) != 0, "Ends list is empty."
            
            # # --- TEST PRINTING ---
            # if len(embeddings) > 0:
            #     # Check shape of first chunk
            #     first_emb = embeddings[0]
            #     print(f"    -> Chunks generated: {len(embeddings)}")
            #     print(f'    -> Processing sequence ID: {id[0]} with length {len(sequence[0])}')
            #     print(f"    -> Start indices: {starts} with length {len(starts)}")
            #     print(f"    -> End indices: {ends} with length {len(ends)}")
            #     print(f"    -> Shape: {first_emb.shape}") 
            #     print(f"    -> First 5 values: {first_emb[0].tolist()}...") # Print a snippet
            # else:
            #     print(f"    -> NO EMBEDDINGS GENERATED")
            # print("-" * 50)
            # # ---------------------
            
            # We took item at 0 because id is a list. 
            # Our fasta files contains only one genome each, so the lists will containonly one element
            for _ in range(len(embeddings)):
                all_ids.append(id[0])
            all_embeddings.extend(embeddings)
            if int(args.summation) == 0:
                all_starts.extend(starts)
                all_ends.extend(ends)

            # if index > 0 and index % 5 == 0:
            #     save_embeddings(
            #         all_ids, 
            #         all_embeddings, 
            #         output_path, 
            #         args.summation,
            #         all_starts if args.summation == 0 else None,
            #         all_ends if args.summation == 0 else None
            #     )

            # Save progress every 200 iterations
            # if index % 200 == 0:
            #     if int(args.summation) == int(1):
            #         # save_embeddings(all_ids, all_embeddings, output_path, args.summation)
            #         save_embeddings(
            #         all_ids, 
            #         all_embeddings, 
            #         output_path, 
            #         args.summation,
            #         all_starts if args.summation == 0 else None,
            #         all_ends if args.summation == 0 else None
            #         )
            #     else:
            #         # save_embeddings(all_ids, all_embeddings, output_path, args.summation, all_starts, all_ends)
            #         save_embeddings(
            #         all_ids, 
            #         all_embeddings, 
            #         output_path, 
            #         args.summation,
            #         all_starts if args.summation == 0 else None,
            #         all_ends if args.summation == 0 else None
            #         )
                    

    except Exception as e:
        # Save right before crashing
        if int(args.summation) == int(1):
            save_embeddings(all_ids, all_embeddings, output_path, args.summation)
        else:
            save_embeddings(all_ids, all_embeddings, output_path, args.summation, all_starts, all_ends)
        raise e

    # Sanity check
    assert(len(all_ids) == len(all_embeddings)), "Error: length mismatch between IDs and Embeddings counts"

    # Stack and convert to numpy array, together with IDs
    # Resulting shape: [N_sequences, 4096]
    embedding_matrix = torch.stack(all_embeddings).numpy() 
    id_array = np.array(all_ids)

    if int(args.summation) == int(1):
        # save_embeddings(all_ids, all_embeddings, output_path, args.summation, all_starts, all_ends)
        save_embeddings(
                    all_ids, 
                    all_embeddings, 
                    output_path, 
                    args.summation,
                    all_starts if args.summation == 0 else None,
                    all_ends if args.summation == 0 else None
        )
    else:
        # save_embeddings(all_ids, all_embeddings, output_path, args.summation, all_starts, all_ends)
        save_embeddings(
                    all_ids, 
                    all_embeddings, 
                    output_path, 
                    args.summation,
                    all_starts if args.summation == 0 else None,
                    all_ends if args.summation == 0 else None
        )