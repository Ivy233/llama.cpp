#!/usr/bin/env python3
import numpy as np
import os
import sys
import glob
import re

def read_embedding_file(filepath):
    """Read embedding file, support different formats"""
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        
        # Try different separators
        if ',' in content:
            # Comma separated
            values = [float(x.strip()) for x in content.split(',') if x.strip()]
        else:
            # Space separated
            values = [float(x) for x in content.split() if x.strip()]
        
        return np.array(values)
    except Exception as e:
        print(f"Error: Cannot read file {filepath}: {e}")
        return None

def calculate_differences(arr1, arr2):
    """Calculate difference statistics between two arrays"""
    if len(arr1) != len(arr2):
        print(f"Warning: Array lengths differ - arr1: {len(arr1)}, arr2: {len(arr2)}")
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
    
    # Absolute differences
    abs_diff = np.abs(arr1 - arr2)
    
    # Relative differences (avoid division by zero)
    rel_diff = np.divide(abs_diff, np.abs(arr1), out=np.zeros_like(abs_diff), where=np.abs(arr1)!=0)
    
    # ✅ New: More reasonable relative error calculation
    arr1_norm = np.linalg.norm(arr1)
    arr2_norm = np.linalg.norm(arr2)
    norm_rel_diff = np.linalg.norm(arr1 - arr2) / arr1_norm * 100
    
    # ✅ New: Filter out relative errors for very small values
    significant_mask = np.abs(arr1) > 1e-6
    significant_rel_diff = rel_diff[significant_mask]
    
    # ✅ Fixed: Correct interval statistics with proper ranges
    num_exact = np.sum(abs_diff == 0)
    num_0_to_1e6 = np.sum((abs_diff > 0) & (abs_diff < 1e-6)) # The missing bin
    num_1e6_to_1e5 = np.sum((abs_diff >= 1e-6) & (abs_diff < 1e-5))
    num_1e5_to_1e4 = np.sum((abs_diff >= 1e-5) & (abs_diff < 1e-4))
    num_1e4_to_1e3 = np.sum((abs_diff >= 1e-4) & (abs_diff < 1e-3))
    num_1e3_to_1e2 = np.sum((abs_diff >= 1e-3) & (abs_diff < 1e-2))
    num_1e2_to_1e1 = np.sum((abs_diff >= 1e-2) & (abs_diff < 1e-1))
    num_above_1e1 = np.sum(abs_diff >= 1e-1)
    
    # Verify total
    total_elements = (num_exact + num_0_to_1e6 + num_1e6_to_1e5 + num_1e5_to_1e4 +
                     num_1e4_to_1e3 + num_1e3_to_1e2 + num_1e2_to_1e1 + num_above_1e1)
    if total_elements != len(arr1):
        # Provide more debug info if assertion fails
        print(f"Warning: Statistics don't add up: {total_elements} != {len(arr1)}")
        uncounted = len(arr1) - total_elements
        print(f"There are {uncounted} uncounted elements.")

    
    # Statistics
    stats = {
        'length': len(arr1),
        'max_abs_diff': np.max(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'median_abs_diff': np.median(abs_diff),
        'max_rel_diff': np.max(rel_diff) * 100,
        'mean_rel_diff': np.mean(rel_diff) * 100,
        'median_rel_diff': np.median(rel_diff) * 100,
        'norm_rel_diff': norm_rel_diff,
        'significant_mean_rel_diff': np.mean(significant_rel_diff) * 100 if len(significant_rel_diff) > 0 else 0,
        'significant_median_rel_diff': np.median(significant_rel_diff) * 100 if len(significant_rel_diff) > 0 else 0,
        # ✅ Fixed statistics with proper ranges
        'num_exact_matches': num_exact,
        'num_0_to_1e6': num_0_to_1e6,
        'num_1e6_to_1e5': num_1e6_to_1e5,
        'num_1e5_to_1e4': num_1e5_to_1e4,
        'num_1e4_to_1e3': num_1e4_to_1e3,
        'num_1e3_to_1e2': num_1e3_to_1e2,
        'num_1e2_to_1e1': num_1e2_to_1e1,
        'num_above_1e1': num_above_1e1,
        # Keep cumulative for backward compatibility, now corrected
        'num_close_matches_1e6': np.sum(abs_diff < 1e-6),
        'num_close_matches_1e5': np.sum(abs_diff < 1e-5),
        'num_close_matches_1e4': np.sum(abs_diff < 1e-4),
        'num_close_matches_1e3': np.sum(abs_diff < 1e-3),
    }
    
    return stats, abs_diff, rel_diff

def find_file_pairs(search_path=None):
    """Find file pairs in specified directory or current directory and subdirectories"""
    if search_path:
        if not os.path.isdir(search_path):
            print(f"Error: Provided search path '{search_path}' is not a valid directory.")
            return []
        all_dirs = [search_path]
        current_dir = os.path.abspath(search_path)
    else:
        current_dir = os.getcwd()
        all_dirs = [current_dir] + [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
    
    all_pairs = []
    
    for dir_path in all_dirs:
        # Find cpp and py files in this directory
        cpp_files = glob.glob(os.path.join(dir_path, "cpp_*_embd.txt"))
        py_files = glob.glob(os.path.join(dir_path, "py_*_embd.txt"))
        
        # Extract format types
        cpp_formats = {}
        for cpp_file in cpp_files:
            match = re.search(r'cpp_(\w+)_embd\.txt', os.path.basename(cpp_file))
            if match:
                format_type = match.group(1)
                cpp_formats[format_type] = cpp_file
        
        py_formats = {}
        for py_file in py_files:
            match = re.search(r'py_(\w+)_embd\.txt', os.path.basename(py_file))
            if match:
                format_type = match.group(1)
                py_formats[format_type] = py_file
        
        # Find matching file pairs in this directory
        for format_type in cpp_formats:
            if format_type in py_formats:
                # ✅ Modified: Include directory information
                dir_name = os.path.basename(dir_path) if dir_path != current_dir else "root"
                all_pairs.append({
                    'format': format_type,
                    'directory': dir_name,
                    'cpp_file': cpp_formats[format_type],
                    'py_file': py_formats[format_type]
                })
    
    return all_pairs

def compare_embeddings(file1, file2, format_type="", directory=""):
    """Compare two embedding files"""
    if format_type and directory:
        print(f"=== BGE-VL Embedding Comparison Tool - {directory.upper()} / {format_type.upper()} Format ===\n")
    elif format_type:
        print(f"=== BGE-VL Embedding Comparison Tool - {format_type.upper()} Format ===\n")
    else:
        print("=== BGE-VL Embedding Comparison Tool ===\n")
    
    # Check if files exist
    if not os.path.exists(file1):
        print(f"Error: File does not exist - {file1}")
        return False
    if not os.path.exists(file2):
        print(f"Error: File does not exist - {file2}")
        return False
    
    print(f"Comparing:")
    print(f"  File1 (llama.cpp): {os.path.basename(file1)}")
    print(f"  File2 (Python):    {os.path.basename(file2)}")
    if directory and directory != "root":
        print(f"  Directory:         {directory}")
    print()
    
    # Read data
    emb1 = read_embedding_file(file1)
    emb2 = read_embedding_file(file2)
    
    print(f'len of emb1: {len(emb1)}')
    print(f'len of emb2: {len(emb2)}')
    
    if emb1 is None or emb2 is None:
        print("Cannot read files, exiting comparison")
        return False
    
    print(f"Data dimensions:")
    print(f"  llama.cpp: {len(emb1)}")
    print(f"  Python:    {len(emb2)}\n")
    
    # Show first 10 values
    print("First 10 values comparison:")
    print("llama.cpp:", emb1[:10])
    print("Python:   ", emb2[:10])
    print()
    
    # Calculate differences
    stats, abs_diff, rel_diff = calculate_differences(emb1, emb2)
    
    # Output statistics
    print("=== Difference Statistics ===")
    print(f"Vector length: {stats['length']}")
    print(f"Exact matches: {stats['num_exact_matches']} ({stats['num_exact_matches']/stats['length']*100:.2f}%)")
    print()
    
    print("Absolute differences:")
    print(f"  Max difference: {stats['max_abs_diff']:.8f}")
    print(f"  Mean difference: {stats['mean_abs_diff']:.8f}")
    print(f"  Median difference: {stats['median_abs_diff']:.8f}")
    print()
    
    print("Relative differences (percentage):")
    print(f"  Max relative difference: {stats['max_rel_diff']:.4f}%")
    print(f"  Mean relative difference: {stats['mean_rel_diff']:.4f}%")
    print(f"  Median relative difference: {stats['median_rel_diff']:.4f}%")
    print(f"  ✅ Norm-based relative error: {stats['norm_rel_diff']:.4f}%")
    print(f"  ✅ Significant values mean relative error: {stats['significant_mean_rel_diff']:.4f}%")
    print()
    
    print("Closeness statistics (by intervals):")
    print(f"  Exact matches (diff = 0): {stats['num_exact_matches']} ({stats['num_exact_matches']/stats['length']*100:.2f}%)")
    print(f"  Difference 0 to 1e-6:   {stats['num_0_to_1e6']} ({stats['num_0_to_1e6']/stats['length']*100:.2f}%)")
    print(f"  Difference 1e-6 to 1e-5: {stats['num_1e6_to_1e5']} ({stats['num_1e6_to_1e5']/stats['length']*100:.2f}%)")
    print(f"  Difference 1e-5 to 1e-4: {stats['num_1e5_to_1e4']} ({stats['num_1e5_to_1e4']/stats['length']*100:.2f}%)")
    print(f"  Difference 1e-4 to 1e-3: {stats['num_1e4_to_1e3']} ({stats['num_1e4_to_1e3']/stats['length']*100:.2f}%)")
    print(f"  Difference 1e-3 to 1e-2: {stats['num_1e3_to_1e2']} ({stats['num_1e3_to_1e2']/stats['length']*100:.2f}%)")
    print(f"  Difference 1e-2 to 1e-1: {stats['num_1e2_to_1e1']} ({stats['num_1e2_to_1e1']/stats['length']*100:.2f}%)")
    print(f"  Difference >= 1e-1: {stats['num_above_1e1']} ({stats['num_above_1e1']/stats['length']*100:.2f}%)")
    print(f"  Total: {stats['length']} (100.00%)")
    print()
    
    # Find positions with largest differences
    max_diff_indices = np.argsort(abs_diff)[-5:][::-1]  # top 5 differences
    print("Top 5 largest differences:")
    for i, idx in enumerate(max_diff_indices):
        print(f"  Position {idx}: llama.cpp={emb1[idx]:.8f}, Python={emb2[idx]:.8f}, diff={abs_diff[idx]:.8f} ({rel_diff[idx]*100:.4f}%)")
    print()
    
    # Calculate cosine similarity
    from numpy.linalg import norm
    cos_sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    print(f"Cosine similarity: {cos_sim:.8f}")
    
    # Calculate L2 distance
    l2_distance = norm(emb1 - emb2)
    print(f"L2 distance: {l2_distance:.8f}")
    
    # Overall assessment
    print("\n=== Overall Assessment ===")
    if stats['norm_rel_diff'] < 0.01:
        print("✅ Very close (norm-based relative error < 0.01%)")
    elif stats['norm_rel_diff'] < 0.1:
        print("✅ Quite close (norm-based relative error < 0.1%)")
    elif stats['norm_rel_diff'] < 1.0:
        print("⚠️  Some differences (norm-based relative error < 1.0%)")
    else:
        print("❌ Significant differences (norm-based relative error >= 1.0%)")
    
    return True

def main():
    """Main function: automatically detect and compare all file pairs"""
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        search_path = sys.argv[1]
        print(f"🔍 Scanning for embedding file pairs in specified directory: {search_path}...\n")
    else:
        search_path = None
    print("🔍 Scanning for embedding file pairs in current directory and subdirectories...\n")
    
    # Find file pairs
    pairs = find_file_pairs(search_path)
    
    if not pairs:
        print("❌ No matching file pairs found!")
        print("Please ensure current directory or subdirectories have cpp_xxx_embd.txt and py_xxx_embd.txt format files")
        return
    
    print(f"✅ Found {len(pairs)} file pairs:")
    for pair in pairs:
        dir_info = f" ({pair['directory']})" if pair['directory'] != "root" else ""
        print(f"  - {pair['format']}{dir_info}: {os.path.basename(pair['cpp_file'])} vs {os.path.basename(pair['py_file'])}")
    print()
    
    # Compare each file pair
    for i, pair in enumerate(pairs):
        if i > 0:
            print("\n" + "="*80 + "\n")
        
        success = compare_embeddings(pair['cpp_file'], pair['py_file'], pair['format'], pair['directory'])
        if not success:
            print(f"❌ Error comparing {pair['format']} format files in {pair['directory']}")
    
    print(f"\n🎉 Completed comparison of all {len(pairs)} file pairs!")

if __name__ == "__main__":
    # Support command line arguments (backward compatibility)
    if len(sys.argv) >= 3 and not os.path.isdir(sys.argv[1]):
        # Manually specify files
        llama_cpp_file = sys.argv[1]
        python_file = sys.argv[2]
        compare_embeddings(llama_cpp_file, python_file)
    else:
        # Automatically detect all file pairs, with optional search path
        main()
