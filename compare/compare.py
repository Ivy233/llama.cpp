#!/usr/bin/env python3
import numpy as np
import os
import sys
import glob
import re

def read_embedding_file(filepath):
    """读取embedding文件，支持不同格式"""
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        
        # 尝试不同的分隔符
        if ',' in content:
            # 逗号分隔
            values = [float(x.strip()) for x in content.split(',') if x.strip()]
        else:
            # 空格分隔
            values = [float(x) for x in content.split() if x.strip()]
        
        return np.array(values)
    except Exception as e:
        print(f"错误: 无法读取文件 {filepath}: {e}")
        return None

def calculate_differences(arr1, arr2):
    """计算两个数组的差异统计"""
    if len(arr1) != len(arr2):
        print(f"警告: 数组长度不同 - arr1: {len(arr1)}, arr2: {len(arr2)}")
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
    
    # 绝对差异
    abs_diff = np.abs(arr1 - arr2)
    
    # 相对差异 (避免除零)
    rel_diff = np.divide(abs_diff, np.abs(arr1), out=np.zeros_like(abs_diff), where=np.abs(arr1)!=0)
    
    # 统计信息
    stats = {
        'length': len(arr1),
        'max_abs_diff': np.max(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'median_abs_diff': np.median(abs_diff),
        'max_rel_diff': np.max(rel_diff) * 100,  # 转换为百分比
        'mean_rel_diff': np.mean(rel_diff) * 100,
        'median_rel_diff': np.median(rel_diff) * 100,
        'num_exact_matches': np.sum(abs_diff == 0),
        'num_close_matches_1e6': np.sum(abs_diff < 1e-6),
        'num_close_matches_1e5': np.sum(abs_diff < 1e-5),
        'num_close_matches_1e4': np.sum(abs_diff < 1e-4),
        'num_close_matches_1e3': np.sum(abs_diff < 1e-3),
    }
    
    return stats, abs_diff, rel_diff

def find_file_pairs():
    """查找当前目录下的文件对"""
    current_dir = os.getcwd()
    cpp_files = glob.glob(os.path.join(current_dir, "cpp_*_embd.txt"))
    py_files = glob.glob(os.path.join(current_dir, "py_*_embd.txt"))
    
    # 提取格式类型
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
    
    # 找到匹配的文件对
    pairs = []
    for format_type in cpp_formats:
        if format_type in py_formats:
            pairs.append({
                'format': format_type,
                'cpp_file': cpp_formats[format_type],
                'py_file': py_formats[format_type]
            })
    
    return pairs

def compare_embeddings(file1, file2, format_type=""):
    """比较两个embedding文件"""
    if format_type:
        print(f"=== BGE-VL Embedding 比较工具 - {format_type.upper()} 格式 ===\n")
    else:
        print("=== BGE-VL Embedding 比较工具 ===\n")
    
    # 检查文件是否存在
    if not os.path.exists(file1):
        print(f"错误: 文件不存在 - {file1}")
        return False
    if not os.path.exists(file2):
        print(f"错误: 文件不存在 - {file2}")
        return False
    
    print(f"正在比较:")
    print(f"  文件1 (llama.cpp): {os.path.basename(file1)}")
    print(f"  文件2 (Python):    {os.path.basename(file2)}\n")
    
    # 读取数据
    emb1 = read_embedding_file(file1)
    emb2 = read_embedding_file(file2)
    
    if emb1 is None or emb2 is None:
        print("无法读取文件，退出比较")
        return False
    
    print(f"数据维度:")
    print(f"  llama.cpp: {len(emb1)}")
    print(f"  Python:    {len(emb2)}\n")
    
    # 显示前10个值
    print("前10个值比较:")
    print("llama.cpp:", emb1[:10])
    print("Python:   ", emb2[:10])
    print()
    
    # 计算差异
    stats, abs_diff, rel_diff = calculate_differences(emb1, emb2)
    
    # 输出统计结果
    print("=== 差异统计 ===")
    print(f"向量长度: {stats['length']}")
    print(f"完全匹配的元素数量: {stats['num_exact_matches']} ({stats['num_exact_matches']/stats['length']*100:.2f}%)")
    print()
    
    print("绝对差异:")
    print(f"  最大差异: {stats['max_abs_diff']:.8f}")
    print(f"  平均差异: {stats['mean_abs_diff']:.8f}")
    print(f"  中位差异: {stats['median_abs_diff']:.8f}")
    print()
    
    print("相对差异 (百分比):")
    print(f"  最大相对差异: {stats['max_rel_diff']:.4f}%")
    print(f"  平均相对差异: {stats['mean_rel_diff']:.4f}%")
    print(f"  中位相对差异: {stats['median_rel_diff']:.4f}%")
    print()
    
    print("接近程度统计:")
    print(f"  差异 < 1e-6: {stats['num_close_matches_1e6']} ({stats['num_close_matches_1e6']/stats['length']*100:.2f}%)")
    print(f"  差异 < 1e-5: {stats['num_close_matches_1e5']} ({stats['num_close_matches_1e5']/stats['length']*100:.2f}%)")
    print(f"  差异 < 1e-4: {stats['num_close_matches_1e4']} ({stats['num_close_matches_1e4']/stats['length']*100:.2f}%)")
    print(f"  差异 < 1e-3: {stats['num_close_matches_1e3']} ({stats['num_close_matches_1e3']/stats['length']*100:.2f}%)")
    print()
    
    # 找出差异最大的几个位置
    max_diff_indices = np.argsort(abs_diff)[-5:][::-1]  # 最大的5个差异
    print("差异最大的5个位置:")
    for i, idx in enumerate(max_diff_indices):
        print(f"  位置 {idx}: llama.cpp={emb1[idx]:.8f}, Python={emb2[idx]:.8f}, 差异={abs_diff[idx]:.8f} ({rel_diff[idx]*100:.4f}%)")
    print()
    
    # 计算余弦相似度
    from numpy.linalg import norm
    cos_sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    print(f"余弦相似度: {cos_sim:.8f}")
    
    # 计算L2距离
    l2_distance = norm(emb1 - emb2)
    print(f"L2距离: {l2_distance:.8f}")
    
    # 总体评估
    print("\n=== 总体评估 ===")
    if stats['mean_rel_diff'] < 0.01:
        print("✅ 非常接近 (平均相对差异 < 0.01%)")
    elif stats['mean_rel_diff'] < 0.1:
        print("✅ 比较接近 (平均相对差异 < 0.1%)")
    elif stats['mean_rel_diff'] < 1.0:
        print("⚠️  有一定差异 (平均相对差异 < 1.0%)")
    else:
        print("❌ 差异较大 (平均相对差异 >= 1.0%)")
    
    return True

def main():
    """主函数：自动检测并比较所有文件对"""
    print("🔍 正在扫描当前目录下的embedding文件对...\n")
    
    # 查找文件对
    pairs = find_file_pairs()
    
    if not pairs:
        print("❌ 未找到匹配的文件对！")
        print("请确保当前目录下有 cpp_xxx_embd.txt 和 py_xxx_embd.txt 格式的文件")
        return
    
    print(f"✅ 找到 {len(pairs)} 对文件:")
    for pair in pairs:
        print(f"  - {pair['format']}: {os.path.basename(pair['cpp_file'])} vs {os.path.basename(pair['py_file'])}")
    print()
    
    # 比较每对文件
    for i, pair in enumerate(pairs):
        if i > 0:
            print("\n" + "="*80 + "\n")
        
        success = compare_embeddings(pair['cpp_file'], pair['py_file'], pair['format'])
        if not success:
            print(f"❌ 比较 {pair['format']} 格式文件时出错")
    
    print(f"\n�� 完成所有 {len(pairs)} 对文件的比较！")

if __name__ == "__main__":
    # 支持命令行参数（向后兼容）
    if len(sys.argv) >= 3:
        # 手动指定文件
        llama_cpp_file = sys.argv[1]
        python_file = sys.argv[2]
        compare_embeddings(llama_cpp_file, python_file)
    elif len(sys.argv) == 2:
        # 只指定Python文件
        python_file = sys.argv[1]
        print("请同时指定两个文件进行比较")
    else:
        # 自动检测所有文件对
        main()