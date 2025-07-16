import torch
from transformers import AutoModel, AutoProcessor
import numpy as np
import imageio
from PIL import Image
import os
import pathlib
import argparse


def get_file_extension(filename):
    return pathlib.Path(filename).suffix.lower().lstrip('.')


def process_image(model, processor, image_path, output_dir):
    # Create unique output filename based on input
    fileext = get_file_extension(image_path)
    output_filename = f"py_{fileext}_embd.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    # Try with direct array processing first
    try:
        image_array = imageio.imread(image_path)
        print(f"stb_image读取结果: shape={image_array.shape}")
        print(f"前10个像素: {image_array.flatten()[:10]}")
        
        with torch.no_grad():
            outputs = model.encode(images=[image_array])
        print("使用stb_image读取 + BGE-VL官方预处理的测试完成")
    except Exception as e:
        print('预期的调试异常:', str(e))
    
    # Process with standard path method
    with torch.no_grad():
        outputs = model.encode(images=[image_path])
    
    embeddings = np.array(outputs).astype(np.float32)
    print(f"嵌入向量形状: {embeddings.shape}")
    
    # Save embeddings
    np.savetxt(output_path, embeddings, fmt="%.6f")
    print(f"图像嵌入向量已保存到{output_path}")
    
    return embeddings


def process_text(model, processor, text, output_dir):
    # Create output filename
    output_filename = f"py_text_embd.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    with torch.no_grad():
        outputs = model.encode(text=[text])
    
    embeddings = np.array(outputs).astype(np.float32)
    print(f"嵌入向量形状: {embeddings.shape}")
    
    # Save embeddings
    np.savetxt(output_path, embeddings, fmt="%.6f")
    print(f"文本嵌入向量已保存到{output_path}")
    
    return embeddings


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process images or text with BGE-VL model'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['image', 'text'],
        help='Processing mode: "image" or "text"'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input path for image or text string'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="/root/compare",
        help='Directory to save output embeddings'
    )
    
    args = parser.parse_args()
    
    # Load model
    model_name = "/root/autodl-tmp/Model/BGE-VL-large"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model.set_processor(model_name)
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process based on mode
    if args.mode == 'image':
        embeddings = process_image(
            model, processor, args.input, args.output_dir
        )
    else:  # args.mode == 'text'
        embeddings = process_text(
            model, processor, args.input, args.output_dir
        )
    
    return embeddings


if __name__ == "__main__":
    main()
