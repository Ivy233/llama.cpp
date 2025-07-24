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

def compare_embeddings_with_stats(file1, file2, format_type="", directory=""):
    """Compare two embedding files and return statistics"""
    # 先运行原来的比较函数显示详细信息
    success = compare_embeddings(file1, file2, format_type, directory)
    
    if not success:
        return None
    
    # 重新读取数据计算统计信息
    emb1 = read_embedding_file(file1)
    emb2 = read_embedding_file(file2)
    
    if emb1 is None or emb2 is None:
        return None
    
    # 计算统计数据
    stats, abs_diff, rel_diff = calculate_differences(emb1, emb2)
    
    # 计算余弦相似度和L2距离
    from numpy.linalg import norm
    cos_sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    l2_distance = norm(emb1 - emb2)
    
    return {
        'format': format_type,
        'directory': directory,
        'cos_sim': cos_sim,
        'l2_dist': l2_distance,
        'norm_rel_err': stats['norm_rel_diff'],
        'max_abs_diff': stats['max_abs_diff'],
        'mean_abs_diff': stats['mean_abs_diff'],
        'exact_matches': stats['num_exact_matches'],
        'total_elements': stats['length']
    }

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
    
    # 统计结果
    results = []
    successful_comparisons = 0
    failed_comparisons = 0
    
    # Compare each file pair
    for i, pair in enumerate(pairs):
        if i > 0:
            print("\n" + "="*80 + "\n")
        
        result = compare_embeddings_with_stats(pair['cpp_file'], pair['py_file'], pair['format'], pair['directory'])
        if result:
            results.append(result)
            successful_comparisons += 1
        else:
            print(f"❌ Error comparing {pair['format']} format files in {pair['directory']}")
            failed_comparisons += 1
    
    # 输出统计报告
    print("\n" + "="*80)
    print("📊 FINAL STATISTICS REPORT")
    print("="*80)
    
    if results:
        print(f"Total tests: {len(results) + failed_comparisons}")
        print(f"Successful tests: {successful_comparisons}")
        print(f"Failed tests: {failed_comparisons}")
        print()
        
        # 余弦相似度统计
        cos_sims = [r['cos_sim'] for r in results if not np.isnan(r['cos_sim'])]
        if cos_sims:
            print("Cosine Similarity Statistics:")
            print(f"  Average: {np.mean(cos_sims):.6f}")
            print(f"  Median:  {np.median(cos_sims):.6f}")
            print(f"  Min:     {np.min(cos_sims):.6f}")
            print(f"  Max:     {np.max(cos_sims):.6f}")
            print()
        
        # 将结果分为文本和图像测试
        text_results = []
        image_results = []
        
        for result in results:
            format_name = result['format'].lower()
            # 判断是否为图像测试（包含 jpg, png, jpeg 等图像格式）
            if any(img_format in format_name for img_format in ['jpg', 'png', 'jpeg', 'gelu', 'quick_gelu']):
                image_results.append(result)
            else:
                text_results.append(result)
        
        # 按数字顺序排序的函数
        def extract_number_from_format(format_str):
            """从格式字符串中提取数字，例如 'text_0' -> 0, 'text' -> 0"""
            import re
            match = re.search(r'(\d+)', format_str)
            return int(match.group(1)) if match else 0
        
        # 分别显示文本和图像测试结果
        if text_results:
            text_results.sort(key=lambda x: extract_number_from_format(x['format']))
            print("📝 TEXT EMBEDDING TESTS:")
            print("-" * 100)
            print(f"{'Test Name':<25} {'Cos Sim':<12} {'L2 Dist':<12} {'Norm Rel Err':<12} {'Status':<15}")
            print("-" * 100)
            
            text_excellent = text_good = text_poor = text_nan = 0
            for result in text_results:
                test_name = f"{result['format']}_{result['directory']}"[:24]
                cos_sim = result['cos_sim']
                l2_dist = result['l2_dist']
                norm_rel_err = result['norm_rel_err']
                
                if np.isnan(cos_sim):
                    status = "❌ NaN"
                    text_nan += 1
                elif cos_sim >= 0.99:
                    status = "✅ Excellent"
                    text_excellent += 1
                elif cos_sim >= 0.95:
                    status = "🟡 Good"
                    text_good += 1
                else:
                    status = "❌ Poor"
                    text_poor += 1
                
                print(f"{test_name:<25} {cos_sim:<12.8f} {l2_dist:<12.6f} {norm_rel_err:<12.4f} {status:<15}")
            
            print("-" * 100)
            print(f"Text Tests Summary: ✅ {text_excellent} excellent, 🟡 {text_good} good, ❌ {text_poor} poor, ❌ {text_nan} NaN")
            print()
        
        if image_results:
            image_results.sort(key=lambda x: extract_number_from_format(x['format']))
            print("🖼️  IMAGE EMBEDDING TESTS:")
            print("-" * 100)
            print(f"{'Test Name':<25} {'Cos Sim':<12} {'L2 Dist':<12} {'Norm Rel Err':<12} {'Status':<15}")
            print("-" * 100)
            
            image_excellent = image_good = image_poor = image_nan = 0
            for result in image_results:
                test_name = f"{result['format']}_{result['directory']}"[:24]
                cos_sim = result['cos_sim']
                l2_dist = result['l2_dist']
                norm_rel_err = result['norm_rel_err']
                
                if np.isnan(cos_sim):
                    status = "❌ NaN"
                    image_nan += 1
                elif cos_sim >= 0.99:
                    status = "✅ Excellent"
                    image_excellent += 1
                elif cos_sim >= 0.95:
                    status = "🟡 Good"
                    image_good += 1
                else:
                    status = "❌ Poor"
                    image_poor += 1
                
                print(f"{test_name:<25} {cos_sim:<12.8f} {l2_dist:<12.6f} {norm_rel_err:<12.4f} {status:<15}")
            
            print("-" * 100)
            print(f"Image Tests Summary: ✅ {image_excellent} excellent, 🟡 {image_good} good, ❌ {image_poor} poor, ❌ {image_nan} NaN")
            print()
        
        # 重新计算总体统计
        excellent_tests = [r for r in results if not np.isnan(r['cos_sim']) and r['cos_sim'] >= 0.99]
        good_tests = [r for r in results if not np.isnan(r['cos_sim']) and 0.95 <= r['cos_sim'] < 0.99]
        problematic_tests = [r for r in results if not np.isnan(r['cos_sim']) and r['cos_sim'] < 0.95]
        nan_tests = [r for r in results if np.isnan(r['cos_sim'])]
        
        # 问题分析
        print("Problem Analysis:")
        print(f"✅ Excellent tests (cos_sim >= 0.99): {len(excellent_tests)}")
        print(f"🟡 Good tests (0.95 <= cos_sim < 0.99): {len(good_tests)}")
        print(f"❌ Poor tests (cos_sim < 0.95): {len(problematic_tests)}")
        print(f"❌ NaN/Error tests: {len(nan_tests)}")
        print()
        
        if problematic_tests:
            print("🔍 Problematic Tests (need attention):")
            for result in problematic_tests:
                print(f"  - {result['format']} ({result['directory']}): cos_sim = {result['cos_sim']:.6f}")
            print()
        
        if nan_tests:
            print("🚨 Tests with NaN/Error (critical issues):")
            for result in nan_tests:
                print(f"  - {result['format']} ({result['directory']}): likely zero vectors or computation error")
            print()
    
    else:
        print("No successful comparisons to report.")
    
    print(f"🎉 Completed analysis of all {len(pairs)} test cases!")

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
