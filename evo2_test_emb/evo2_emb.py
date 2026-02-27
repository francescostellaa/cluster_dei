import torch
from evo2 import Evo2
import os
import Bio.SeqIO as SeqIO
from torch.utils.data import DataLoader, Dataset
import argparse
import numpy as np
from datetime import datetime
import itertools
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path

# Global variable for debug purposes
VERBOSE = False


def filter_protein_regions(sequence: str, gbk_file, keep_known_proteins=False, tail_erase_size=0):
    """
    Filters a DNA sequence based on CDS features in a GenBank file.
    
    Args:
        sequence (str): The input DNA sequence.
        gbk_file (str): Path to the GenBank file.
        keep_known_proteins (bool): 
            If False (default): Removes known proteins (keeps hypothetical/junk).
            If True: Keeps ONLY known proteins (removes hypothetical/junk).
        tail_erase_size (int): Number of bases to remove from the start and end of known protein regions (default: 0).
    """
    seq_len = len(sequence)
    
    # Start with a mask where True = "This is a Known Protein"
    # Initialize to False (assume nothing is a known protein yet)
    known_protein_mask = np.zeros(seq_len, dtype=bool)

    try:
        if VERBOSE:
            print(f"Original sequence length: {seq_len}")
            print(f"Mode: {'KEEP only known proteins' if keep_known_proteins else 'REMOVE known proteins'}")

        # Parse GenBank to find known proteins
        for record in SeqIO.parse(gbk_file, "genbank"):
            for feature in record.features:
                if feature.type == "CDS":
                    product = feature.qualifiers.get("product", ["None"])[0].lower().strip()
                    
                    # Logic: Identify "Known Proteins"
                    if product and "hypothetical protein" not in product:
                        start = int(feature.location.start)
                        end = int(feature.location.end)
                        
                        # Bounds check
                        start = max(0, start)
                        end = min(seq_len, end)
                        
                        if start < end:
                            # Mark this region as a Known Protein.
                            # If tail_erase_size > 0, then enlarge the masked region if possible
                            if tail_erase_size > 0:
                                start = max(0, start - tail_erase_size)
                                end = min(seq_len, end + tail_erase_size)
                            known_protein_mask[start:end] = True

        # Apply the Flag Logic
        if keep_known_proteins:
            # We want ONLY the known proteins. The mask is already correct.
            final_mask = known_protein_mask
        else:
            # We want everything EXCEPT known proteins. Invert the mask.
            final_mask = ~known_protein_mask

        # Apply mask to sequence
        seq_array = np.array(list(sequence))
        cleaned_sequence = "".join(seq_array[final_mask])

        if VERBOSE:
            print(f"Final sequence length: {len(cleaned_sequence)}")
        
        return str(cleaned_sequence)

    except Exception as e:
        print(f"Warning: Failed to process GBK file {gbk_file}. Error: {e}")
        return str(sequence)    
    
    
    
    

def remove_known_proteins(sequence, gbk_file):
    """
    Scans a GBK file for CDS features. if the product is NOT 'hypothetical protein',
    it removes that region from the sequence.
    
    Args:
        sequence (str): The input DNA sequence (from FASTA).
        gbk_file (str): Path to the corresponding GenBank file.
        
    Returns:
        str: The sequence with known protein regions removed.
    """
    # Create a boolean mask of the same length as the sequence (True = keep, False = remove)
    seq_len = len(sequence)
    mask = np.ones(seq_len, dtype=bool)
    if VERBOSE:
        print(f"Length of sequence with known proteins: {seq_len}")
    try:
        # Parse the GenBank file
        # We assume the GenBank sequence aligns with the FASTA sequence
        for record in SeqIO.parse(gbk_file, "genbank"):
            for feature in record.features:
                if feature.type == "CDS":
                    # Get the product name safely
                    product = feature.qualifiers.get("product", ["None"])[0].lower().strip()
                    
                    # Logic: If it is NOT a hypothetical protein, we treat it as "known" and remove it
                    if product and "hypothetical protein" not in product:
                        # BioPython locations are 0-based, exact slice indices
                        start = int(feature.location.start)
                        end = int(feature.location.end)
                        
                        # Ensure we don't go out of bounds if FASTA/GBK lengths differ slightly
                        start = max(0, start)
                        end = min(seq_len, end)
                        
                        # Mark this region for removal
                        if start < end:
                            mask[start:end] = False
                            
        # Reconstruct the sequence keeping only the True positions
        seq_array = np.array(list(sequence))
        cleaned_sequence = "".join(seq_array[mask])
        if VERBOSE:
            print(f"Length of sequence without known proteins: {len(cleaned_sequence)}")
        
        return str(cleaned_sequence)

    except Exception as e:
        print(f"Warning: Failed to process GBK file {gbk_file}. Using original sequence. Error: {e}")
        return str(sequence)



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
    # embeddings = []
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


def save_embeddings(all_ids, all_embeddings, output_path, summation, all_strands, all_starts=None, all_ends=None ):
    """
    Store data to specified output path.
    Args:
        all_ids (List): list of ids
        all_embeddigns (List): list of embeddings
        output_path (str): string that represent the path where ids and embeddings has to be saved 
        summation (int): Flag to know if the methods needs to save starts and ends
        all_stars (List): List of all start inidices for the embeddings
        all_ends (List): Llist of all end indices for the embeddings
        all_strands (List): List of all strands for the embeddings
    Returns:
        None
    Raise:
        Error if something goes wrong while saving operation
    """
    assert len(all_ids) == len(all_embeddings), "Length mismatch between IDs and Embeddings counts"
    assert len(all_strands) == len(all_embeddings), "Length mismatch between Strands and Embeddings counts"
    
    try:
        if int(summation) == int(1):
            torch.save(
                {
                    "ids": all_ids,
                    "embeddings": all_embeddings,
                    "strands": all_strands
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
                    "strands": all_strands
                },
                output_path
            )
            
    except Exception as e:
        print("Error while saving data!")
        raise e
    return

def opposite_strand(sequence: str) -> str:
    """
    Generate the opposite strand of a given DNA sequence.
    Args:
        sequence (str): The input DNA sequence.
    Returns:
        str: The opposite strand of the input sequence.
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    opposite = ''.join(complement.get(base, base) for base in reversed(sequence))
    return opposite



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
    parser.add_argument("--strands", type=int, default=0, dest="strands",
                        help="If parameter is 1, embedding for both strands will be computed, otherwise only provided one (default: 0)")
    parser.add_argument("--gbk", dest="gbk_dir",
                        help="Input directory containing GenBank files", metavar="DIR")
    parser.add_argument("--keep_known_proteins", action="store_true", dest="keep_known_proteins",
                        help="If set, keeps ONLY known proteins and discards the rest."
                        "If omitted (default), removes known proteins and keeps the rest.")
    parser.add_argument("--tail_erase_size", type=int, default=0, dest="tail_erase_size",
                        help="Number of bases to remove from the start and end of known protein regions (default: 0, no tail erasing)")
    # Parse arguments
    args = parser.parse_args()
    
    if args.keep_known_proteins and not args.gbk_dir:
        parser.error("--keep_known_proteins requires --gbk to be specified.")
    
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
    all_strands = [] # List of all strands for each embeddings
        
        
    date = datetime.now()
    # String format: date_time_dataset-length_window-size_layer-name_stride
    output_path = os.path.join(
        args.output_dir, 
        f'embeddings_{date.year}-{date.month}-{date.day}_{date.hour}-{date.minute}-{date.second}_{len(dataset)}_{args.window_size}_{args.layer_names}_{args.stride}_{args.summation}_{args.strands}.pt'
    )
    
    TEST_LIMIT = 5
    
    try:
        for index, pair in enumerate(dataloader):

            # TEST: Stop loop if we exceed the limit
            if VERBOSE:
                if TEST_LIMIT and index >= TEST_LIMIT:
                    print(f"\n--- Test limit of {TEST_LIMIT} reached. Stopping loop. ---", flush=True)
                    break
            
            # We need to clear cache every n iterations, otherwise. 
            if index % 500 == 0:
                torch.cuda.empty_cache()

            id, sequence = pair
            sequence = str(sequence[0])
            # List of embeddings, starting and end positions for current sequence
            embeddings, starts, ends, strands = [], [], [], []

            if args.gbk_dir:
                # Assume GBK filename matches the FASTA ID
                gbk_file = os.path.join(args.gbk_dir, f"{(id[0].split('.')[0]).strip()}.gbk")
                if os.path.exists(gbk_file):
                    # Filter the sequence
                    # sequence = remove_known_proteins(sequence, gbk_file)   
                    sequence = filter_protein_regions(sequence, gbk_file, args.keep_known_proteins, args.tail_erase_size)
                    if len(sequence) == 0:
                        print(f"Skipping {id[0]}: Empty after filtering.")
                        continue

            get_embedding(
                model,
                sequence,
                args.layer_names,
                args.window_size,
                args.stride,
                args.summation,
                embeddings,
                starts,
                ends
            )
            # Extend strands
            strands.extend(itertools.repeat(1, len(embeddings)))
            
            if int(args.strands) == 1:
                # Generate opposite strand
                opp_sequence = opposite_strand(sequence)
                get_embedding(
                    model,
                    opp_sequence,
                    args.layer_names,
                    args.window_size,
                    args.stride,
                    args.summation,
                    embeddings,
                    starts,
                    ends
                )
                # Add strands values for opposite strand. Add for half the length of embeddings list.
                # Half because we already added the first half for the original strand
                count = len(embeddings) // 2
                strands.extend(itertools.repeat(-1, count))
            
            
            # --- TEST PRINTING ---
            if VERBOSE:
                if len(embeddings) > 0:
                    # Check shape of first chunk
                    first_emb = embeddings[0]
                    print(f"    -> Chunks generated: {len(embeddings)}", flush=True)
                    print(f'    -> Processing sequence ID: {id[0]} with length {len(sequence)}', flush=True)
                    print(f"    -> Start indices: {starts} with length {len(starts)}", flush=True)
                    print(f"    -> End indices: {ends} with length {len(ends)}", flush=True)
                    print(f"    -> Strands: {strands}", flush=True)
                    print(f"    -> Shape: {first_emb.shape}", flush=True) 
                    print(f"    -> First 5 values: {first_emb[0].tolist()}...", flush=True) # Print a snippet
                else:
                    print(f"    -> NO EMBEDDINGS GENERATED")
                print("-" * 50)
            # ---------------------
            
            assert len(embeddings) != 0, "Embeddings list is empty."
            assert len(starts) != 0, "Starts list is empty."
            assert len(ends) != 0, "Ends list is empty."
            assert len(strands) != 0, "Strands list is empty."
            assert len(embeddings) == len(starts) == len(ends) == len(strands), "Length mismatch among embeddings, starts, ends, and strands lists."
            
            # We took item at 0 because id is a list. 
            # Our fasta files contains only one genome each, so the lists will containonly one element
            # for _ in range(len(embeddings)):
            #     all_ids.append(id[0])
            all_ids.extend(itertools.repeat(id[0], len(embeddings)))
            all_embeddings.extend(embeddings)
            if int(args.summation) == 0:
                all_starts.extend(starts)
                all_ends.extend(ends)
                
            all_strands.extend(strands)

            # --- TEST STORING ---
            if VERBOSE:
                if index > 0 and index % 5 == 0:
                    save_embeddings(
                        all_ids, 
                        all_embeddings, 
                        output_path, 
                        args.summation,
                        all_strands,
                        all_starts if args.summation == 0 else None,
                        all_ends if args.summation == 0 else None
                    )        
            # --------------

    except Exception as e:
        # Save right before crashing
        save_embeddings(
                all_ids, 
                all_embeddings, 
                output_path, 
                args.summation,
                all_strands,
                all_starts if args.summation == 0 else None,
                all_ends if args.summation == 0 else None
        )
        raise e

    # Sanity check
    assert(len(all_ids) == len(all_embeddings)), "Error: length mismatch between IDs and Embeddings counts"

    # Stack and convert to numpy array, together with IDs
    # Resulting shape: [N_sequences, 4096]
    embedding_matrix = torch.stack(all_embeddings).numpy() 
    id_array = np.array(all_ids)


    save_embeddings(
                all_ids, 
                all_embeddings, 
                output_path, 
                args.summation,
                all_strands,
                all_starts if args.summation == 0 else None,
                all_ends if args.summation == 0 else None
    )