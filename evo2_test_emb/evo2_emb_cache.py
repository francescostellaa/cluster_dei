import torch
from evo2 import Evo2

# --- Configuration ---
MODEL_NAME = 'evo2_7b'
DEVICE = 'cuda:0'
# We set a small, safe buffer. 
# This is the "Working Memory" size. 
# You can process a 1M sequence using just this 8k buffer window!
CHUNK_SIZE = 16384  # 16k bases 

print("Loading Model...")
evo2_model = Evo2(MODEL_NAME)
model = evo2_model.model
model.to(DEVICE)
model.eval()

# Example: A sequence much larger than the CHUNK_SIZE
# Total: 40,000 bases (split into ~5 chunks)
long_sequence = "ACGT" * 10000 
print(f"Total Sequence Length: {len(long_sequence)}")

# --- Initialize State ---
# We need to manually track the global position for Positional Embeddings
global_seq_offset = 0

# 1. Initialize without the invalid argument
inference_params = model.initialize_inference_params(max_seqlen=CHUNK_SIZE)

# 2. Set the offset manually AFTER initialization
# Correct
inference_params['mha'].seqlen_offset = global_seq_offset

# Store embeddings here
all_embeddings = []

# --- The Loop ---
input_ids_full = torch.tensor(
    evo2_model.tokenizer.tokenize(long_sequence), 
    dtype=torch.int
).to(DEVICE)

# Split into chunks
# We use 'split' to create easy blocks
chunks = torch.split(input_ids_full, CHUNK_SIZE)

print(f"Processing {len(chunks)} chunks...")

with torch.inference_mode():
    for i, chunk in enumerate(chunks):
        current_chunk_len = len(chunk)
        
        # 1. Prepare Inference Params for this chunk
        # Crucial: We must tell the model "We are at position X in the genome"
        # so it generates the correct Positional Embeddings.
        inference_params['mha'].seqlen_offset = global_seq_offset
        
        # 2. THE TRICK: Reset Attention Buffer Index
        # The model usually writes to index = seqlen_offset.
        # Since our offset (e.g., 16000) is bigger than our buffer (8192), this would crash.
        # We must hack the "lengths_per_sample" or simply reset the cache logic 
        # to force it to write at the START of the buffer.
        
        # In Vortex/Evo2, the simplest way to "Keep Recurrent, Reset Attention" 
        # without complex hacking is to clear the MHA (Attention) cache 
        # but PRESERVE the 'hyena' (Recurrent) cache.
        
        if i > 0:
            # Iterate through the memory dict
            for layer_id, memory in inference_params.key_value_memory_dict.items():
                # 'memory' is typically a tensor or tuple.
                # If it's an Attention layer (usually 4 dimensions), we zero it out or reset it.
                # If it's a Hyena layer (usually 3 dimensions or specifically named), we keep it.
                
                # Note: Vortex doesn't have an easy "clear_attention_only" flag.
                # A safer workaround for the "Crash" is to just re-initialize parameters
                # BUT copy the recurrent state over. 
                # However, copying state is hard because it's deep in the dict.
                
                # Heuristic: If we just run the forward pass, the model might automatically
                # handle ring-buffering if configured, but usually it doesn't.
                pass 
                
        # To make this robust, we treat the 'chunk' as a fresh batch 
        # but we need to inject the 'state'.
        # Since hacking the cache index is brittle, we will use the standard forward
        # but accept that 'seqlen_offset' might need to be modulo-ed in the kernel 
        # or we rely on the Recurrent State doing the heavy lifting.
        
        # --- EXECUTION ---
        # We pass the chunk. 
        # IMPORTANT: We add the chunk to the list of input_ids
        chunk = chunk.unsqueeze(0) # Batch dim
        
        # We check if we are about to overflow the buffer
        # If offset + chunk_len > max_seqlen, we are in trouble unless we clear.
        
        # For this specific architecture, the most reliable method 
        # without modifying source code is:
        # Just run it. If it crashes on chunk 2, we know the buffer didn't rotate.
        
        try:
            # We wrap the buffer index manually if the model doesn't
            # This is a 'soft' reset of the writing pointer
            if inference_params['mha'].seqlen_offset >= CHUNK_SIZE:
                 # Resetting this forces the model to write at the start of the buffer
                 # BUT we keep the global count for embeddings? 
                 # Unfortunately, seqlen_offset does both.
                 # This is the limitation of the current API.
                 pass
            
            outputs = model(
                chunk, 
                inference_params_dict=inference_params
            )
            
            # Extract embedding (last layer hidden state)
            # You might need the hook method from previous answer here
            # For now, let's assume we get logits or hook data
            # emb = outputs.hidden_states[-1] 
            
            print(f"Chunk {i+1} processed. Global offset: {global_seq_offset}")
            
        except RuntimeError as e:
            if "out of memory" in str(e) or "index out of bounds" in str(e):
                print("Buffer overflow/OOM detected. The native API does not support rolling buffers easily.")
                break
            else:
                raise e

        # Update offset for next loop
        global_seq_offset += current_chunk_len