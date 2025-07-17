import torch
from transformers import AutoModel, AutoProcessor
import numpy as np
import imageio
import os
import pathlib
import argparse


def get_file_extension(filename):
    return pathlib.Path(filename).suffix.lower().lstrip('.')


def process_image(model, processor, image_path, output_dir):
    # Get suffix from environment variable
    suffix = os.getenv("OUTPUT_SUFFIX", "")

    # Create unique output filename based on input
    fileext = get_file_extension(image_path)
    output_filename = f"py_{suffix if suffix else fileext}_embd.txt"
    output_path = os.path.join(output_dir, output_filename)

    # Process with standard path method
    with torch.no_grad():
        outputs = model.encode(images=[image_path])

    embeddings = np.array(outputs).astype(np.float32)
    print(f"嵌入向量形状: {embeddings.shape}")

    # Save embeddings
    np.savetxt(output_path, embeddings, fmt="%.6f")
    print(f"图像嵌入向量已保存到 {output_path}")

    return embeddings


def process_text(model, processor, text, output_dir):
    # Get suffix from environment variable
    suffix = os.getenv("OUTPUT_SUFFIX", "")

    # Create unique output filename based on suffix
    output_filename = f"py_{suffix if suffix else 'text'}_embd.txt"
    output_path = os.path.join(output_dir, output_filename)

    with torch.no_grad():
        print(f'text: {text}')
        outputs = model.encode(text=[text])

    embeddings = np.array(outputs).astype(np.float32)
    print(f"嵌入向量形状: {embeddings.shape}")

    np.savetxt(output_path, embeddings, fmt="%.6f")
    print(f"文本嵌入向量已保存到 {output_path}")

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="获取 BGE-VL 模型在给定输入下的嵌入向量。")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件夹的路径。")
    parser.add_argument("--prompt", type=str, help="要编码的文本。")
    parser.add_argument("--image_path", type=str, help="要编码的图像文件的路径。")

    args = parser.parse_args()

    # Get the output directory from environment variable, default to current dir
    output_dir = os.getenv("OUT_DIR", ".")
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型和处理器
    print("正在加载模型和处理器...")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    print("模型和处理器加载成功。")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.set_processor(args.model_path)
    model.eval()

    if args.prompt:
        process_text(model, processor, args.prompt, output_dir)

    if args.image_path:
        process_image(model, processor, args.image_path, output_dir)

    if not args.prompt and not args.image_path:
        print("错误：必须提供 --prompt 或 --image_path 中的至少一个。")
        parser.print_help()


if __name__ == "__main__":
    main()
