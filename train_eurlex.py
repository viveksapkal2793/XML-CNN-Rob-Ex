import shutil
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime
import torch
import yaml

from build_problem import BuildProblem
from my_functions import out_size

def setup_eurlex_dataset():
    """Set up the Eurlex dataset for XML-CNN"""
    
    # Step 1: Check if data files exist, download if not
    eurlex_train = "eurlex_data/eurlex_train.txt"
    eurlex_test = "eurlex_data/eurlex_test.txt"
    
    if not os.path.exists(eurlex_train) or not os.path.exists(eurlex_test):
        print("Eurlex dataset files not found. Please place them in data/ directory.")
        print("Example format: 446,521,1149,1249 0:0.084556 1:0.138594 2:0.094304...")
        return False
    
    # Step 2: Convert Eurlex format to XML-CNN format
    print("Converting Eurlex format to XML-CNN compatible format...")
    subprocess.call([
        "python", "eurlex_data/convert_eurlex.py", 
        eurlex_train, "eurlex_data/train_org.txt"
    ])
    
    subprocess.call([
        "python", "eurlex_data/convert_eurlex.py", 
        eurlex_test, "eurlex_data/test.txt"
    ])
    
    # Step 3: Create validation set
    print("Creating validation set...")
    subprocess.call(["python", "eurlex_data/make_valid.py", "eurlex_data/train_org.txt"])
    
    # Step 4: Create BOW embeddings
    print("Creating BOW embeddings...")
    subprocess.call(["python", "eurlex_data/create_bow_embeddings.py", "--dim", "5000", "--emb-dim", "300"])
    
    return True

def main():
    parser = ArgumentParser()
    parser.add_argument("--use_cpu", help="Use CPU instead of GPU", action="store_true")
    parser.add_argument("--no_preproc", help="if preprocessing is done", action="store_true")
    args = parser.parse_args()
    
    # Setup Eurlex dataset
    if not args.no_preproc:
        if not setup_eurlex_dataset():
            print("Failed to set up Eurlex dataset. Exiting.")
            return
    
    # Load parameters from config
    with open("params.yml") as f:
        params = yaml.safe_load(f)
    
    common = params["common"]
    hyper_params = params["hyper_params"]
    normal_train = params["normal_train"]
    
    # Update parameters for Eurlex
    normal_train["train_data_path"] = "eurlex_data/train.txt"
    normal_train["valid_data_path"] = "eurlex_data/valid.txt"
    normal_train["test_data_path"] = "eurlex_data/test.txt"
    
    # Use BOW embeddings instead of GloVe
    common["vector_cache"] = ".vector_cache/bow_embeddings.txt"
    common["model_name"] = f"eurlex_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Determine device
    use_device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    
    # Set parameters
    all_params = {
        "device": use_device, 
        "params_search": False
    }
    all_params.update(common)
    all_params.update(hyper_params)
    all_params.update(normal_train)
    
    # Print settings
    term_size = shutil.get_terminal_size().columns
    print("\n" + " Eurlex Training Mode ".center(term_size, "="))
    print("\n" + " Params ".center(term_size, "-"))
    print([i for i in sorted(common.items())])
    print("-" * term_size)
    
    print("\n" + " Hyper Params ".center(term_size, "-"))
    print([i for i in sorted(hyper_params.items())])
    print("-" * term_size)
    
    # Train model
    trainer = BuildProblem(all_params)
    trainer.preprocess()
    trainer.run()

if __name__ == "__main__":
    main()