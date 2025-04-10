import os
import argparse
import numpy as np
from tqdm import tqdm

def convert_eurlex_to_text_format(input_file, output_file, feature_dim=5000, sequence_length=500):
    """
    Convert Eurlex-4k BOW format to a sequence format compatible with XML-CNN
    
    Input format: "446,521,1149,1249 0:0.084556 1:0.138594 2:0.094304..."
    Output format: "446,521,1149,1249\tfeature1 feature2 feature3..."
    
    The idea is to convert sparse features to a "pseudo-text" sequence
    where feature IDs with highest values become tokens in the sequence.
    """
    print(f"Converting {input_file} to {output_file}...")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:

        # Skip the first line with metadata
        first_line = f_in.readline().strip()
        print(f"Skipping metadata line: {first_line}")
        
        for line in tqdm(f_in):
            parts = line.strip().split(' ', 1)
            if len(parts) != 2:
                continue
                
            labels, features_str = parts
            
            # Parse features (sparse format)
            features = {}
            for feature in features_str.split():
                idx, val = feature.split(':')
                features[int(idx)] = float(val)
            
            # Sort features by value (highest first) and take top sequence_length
            sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:sequence_length]
            
            # Create a sequence of feature IDs as tokens
            # We prefix with "f" to make it clear these are feature IDs
            token_seq = " ".join([f"f{idx}" for idx, _ in top_features])
            
            # If we have fewer than sequence_length features, pad with a special token
            if len(top_features) < sequence_length:
                padding = " pad" * (sequence_length - len(top_features))
                token_seq += padding
            
            # Write in the format expected by XML-CNN: ID\tLABELS\tTEXT
            # Using the document index as ID
            f_out.write(f"{labels}\t{token_seq}\n")
    
    print(f"Conversion complete. Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert Eurlex-4k BOW format to XML-CNN compatible format")
    parser.add_argument("input_file", help="Path to input Eurlex-4k file")
    parser.add_argument("output_file", help="Path to output file")
    parser.add_argument("--dim", type=int, default=5000, help="Feature dimension")
    parser.add_argument("--seq-len", type=int, default=500, help="Max sequence length")
    
    args = parser.parse_args()
    
    convert_eurlex_to_text_format(args.input_file, args.output_file, args.dim, args.seq_len)

if __name__ == "__main__":
    main()