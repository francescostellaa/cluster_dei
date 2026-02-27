from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, 
                             matthews_corrcoef)
import numpy as np
import torch
import pandas as pd

# Define function to print metrics for each model
def compute_and_print_metrics(model_name, y_true, y_pred, pos_label="lysogenic"):
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    neg_label = "lytic" if pos_label == "lysogenic" else "lysogenic"
    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n{model_name} RESULTS")
    print("-" * 30)
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"Accuracy:             {acc:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    print(f"MCC:                  {mcc:.4f}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(f"                 Pred {neg_label}  Pred {pos_label}")
    print(f"Actual {neg_label:9} {tn:5d}           {fp:5d}")
    print(f"Actual {pos_label:9} {fn:5d}           {tp:5d}")




# Load the file using the path exactly as it was saved
# file_path = '../data/embeddings_2026-1-10_15-24-10_2176_32768_blocks.28.mlp.l3_16384_0.pt'
# Genome formed only by not codings regions
# file_path = '../data/embeddings_2026-1-27_12-28-7_2176_32768_blocks.28.mlp.l3_16384_0_0.pt'
# Genome formed only by codings regions

def load_data(file_path):
    data = torch.load(file_path, map_location=torch.device('cpu'))

    # Extract the lists from the dictionary
    ids = data["ids"]               # List of strings
    # DeepPL files contains accession id without version number, so we need to erase it from our ids
    ids = [item.split('.')[0] for item in ids] 
    embeddings_list = data["embeddings"] # List of torch.Tensors
    starts_list = data['starts']
    ends_list = data['ends']

    # Stack the list of tensors into one large tensor (N_samples x 4096)
    embeddings_tensor = torch.stack(embeddings_list)

    # Convert to Numpy array
    loaded_embeds = embeddings_tensor.detach().cpu().numpy()
    loaded_ids = np.array(ids)

    # Sanity checks
    assert len(ids) == loaded_embeds.shape[0] == len(starts_list) == len(ends_list)

    print("IDs array shape:", loaded_ids.shape)
    print("Embeddings array shape:", loaded_embeds.shape) 

    df = pd.DataFrame(
        data=loaded_embeds[:,:],
        index=loaded_ids[:],
        columns=[[f"feature_{i}" for i in range(len(loaded_embeds[0]))]]
    ) 
    
    return df, starts_list, ends_list




# Align data
# Path to the Excel file containing lifecycle information
def align_data(df, excel_file_path):
    # Training
    sheet_names = pd.ExcelFile(excel_file_path).sheet_names
    # Training dataset
    training = pd.read_excel(excel_file_path, sheet_name=sheet_names[0])
    # Test dataset
    test = pd.read_excel(excel_file_path, sheet_name=sheet_names[1])

    # Create a dictionary mapping Accession ID -> Corrected Lifecycle
    lifecycle_map = {}
    all_metadata = pd.concat([training, test])

    # Standardize lifecycle labels and populate the map
    for _, row in all_metadata.iterrows():
        acc = row['Accession number']
        raw_life = row['Lifecycle']
        clean_life = "lysogenic" if str(raw_life).strip().lower() in ["lysogneic", "lysogenic"] else "lytic"
        lifecycle_map[acc] = clean_life

    # Create aligned list
    aligned_lifecycles = []
    for accession in df.index:
        if accession in lifecycle_map:
            aligned_lifecycles.append(lifecycle_map[accession])
        else:
            # Warning only once per missing ID to avoid clutter
            print(f"Warning: Accession {accession} not found. Defaulting to lytic.")
            aligned_lifecycles.append("lytic") 

    aligned_lifecycles = np.array(aligned_lifecycles)
    
    train_ids = set(training['Accession number'].astype(str).str.strip())
    test_ids = set(test['Accession number'].astype(str).str.strip())

    # Prepare feature and labels
    train_indices = []
    test_indices = []
    y_train = []
    y_test = []

    for i, accession in enumerate(df.index):
        label = lifecycle_map.get(accession, "lytic")
        
        if accession in train_ids:
            train_indices.append(i)
            y_train.append(label)
        elif accession in test_ids:
            test_indices.append(i)
            y_test.append(label)
    
    return aligned_lifecycles, train_indices, test_indices, y_train, y_test