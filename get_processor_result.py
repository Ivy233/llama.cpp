import torch
from transformers import AutoModel, AutoProcessor
import numpy as np
import os
import pathlib
import argparse
from PIL import Image


def get_file_extension(filename):
    return pathlib.Path(filename).suffix.lower().lstrip('.')


import torch
from transformers import AutoModel, AutoProcessor
import numpy as np
import os
import pathlib
import argparse
from PIL import Image


def get_file_extension(filename):
    return pathlib.Path(filename).suffix.lower().lstrip('.')


def stbi_load_equivalent(image_path, req_comp=3):
    """
    模仿 stbi_load(image_path.c_str(), &width, &height, &channels, 3)
    - 直接从文件路径加载
    - req_comp=3 强制输出RGB格式
    - 返回 uint8 数组 (0-255)
    """
    # 使用PIL模仿STB行为
    with Image.open(image_path) as img:
        # 获取原始尺寸和通道信息
        width, height = img.size
        original_channels = len(img.getbands())
        
        # STB的req_comp=3表示强制转换为RGB
        if req_comp == 3:
            rgb_img = img.convert('RGB')
        else:
            rgb_img = img  # 保持原格式
            
        # 转换为numpy数组，模仿STB的内存布局
        # STB返回的是 unsigned char* 按 RGBRGBRGB... 存储
        rgb_data = np.array(rgb_img, dtype=np.uint8)  # shape: (height, width, 3)
        
    return rgb_data, width, height, original_channels


def process_image_stb_aligned(model, processor, image_path, output_dir):
    """
    完全对齐STB的图片处理方式
    """
    import tempfile
    
    suffix = os.getenv("OUTPUT_SUFFIX", "")
    fileext = get_file_extension(image_path)
    output_filename = f"py_{suffix if suffix else fileext}_embd.txt"
    output_path = os.path.join(output_dir, output_filename)

    print(f"🖼️ 使用STB对齐方式处理: {image_path}")
    
    # 模仿 stbi_load 的精确行为
    rgb_data, width, height, channels = stbi_load_equivalent(image_path, req_comp=3)
    print(f"STB加载结果: {width}x{height}, 原始通道={channels}, 输出通道=3")
    print(f"RGB数据形状: {rgb_data.shape}, 数据类型: {rgb_data.dtype}")
    
    # 将STB处理后的数据保存为临时文件，然后传递路径给模型
    # 这样可以绕过模型内部的PIL处理，确保使用我们的STB数据
    pil_img = Image.fromarray(rgb_data)
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
        pil_img.save(temp_path, 'PNG')
        print(f"STB数据保存到临时文件: {temp_path}")
    
    try:
        # 使用临时文件路径进行编码（模型会重新加载，但我们已经按STB方式预处理过）
        with torch.no_grad():
            outputs = model.encode(images=[temp_path])

        embeddings = np.array(outputs).astype(np.float32)
        print(f"嵌入向量形状: {embeddings.shape}")

        np.savetxt(output_path, embeddings, fmt="%.6f")
        print(f"STB对齐处理完成，保存到: {output_path}")

        return embeddings
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def process_image_original_path(model, processor, image_path, output_dir):
    """
    原始的路径传递方式 (模型内部使用PIL)
    """
    suffix = os.getenv("OUTPUT_SUFFIX", "")
    fileext = get_file_extension(image_path)
    output_filename = f"py_{suffix if suffix else fileext}_embd.txt"
    output_path = os.path.join(output_dir, output_filename)

    print(f"🖼️ 使用原始路径处理: {image_path}")
    
    # 直接传递路径，让模型内部处理
    with torch.no_grad():
        outputs = model.encode(images=[image_path])

    embeddings = np.array(outputs).astype(np.float32)
    print(f"嵌入向量形状: {embeddings.shape}")

    np.savetxt(output_path, embeddings, fmt="%.6f")
    print(f"原始路径处理完成，保存到: {output_path}")

    return embeddings


def process_image(model, processor, image_path, output_dir):
    """
    选择处理方式：通过环境变量控制
    USE_STB_ALIGNED=1 : 使用STB对齐方式
    USE_STB_ALIGNED=0 : 使用原始路径方式 (默认)
    """
    use_stb_aligned = os.getenv("USE_STB_ALIGNED", "0") == "1"
    
    if use_stb_aligned:
        return process_image_stb_aligned(model, processor, image_path, output_dir)
    else:
        return process_image_original_path(model, processor, image_path, output_dir)


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
