#!/bin/bash

# 设置环境变量和目录
export OUT_DIR=/root/tmp/llama.cpp/compare/text_suite
echo "输出目录设置为: $OUT_DIR"

# 清理旧的测试结果
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
echo "已清理并创建输出目录。"

# 编译 C++ embedding 程序
echo "正在编译 C++ embedding 程序..."
cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build --target llama-embedding -j
cmake --build build --target llama-tokenize -j
if [ $? -ne 0 ]; then
    echo "C++ 程序编译失败，脚本终止。"
    exit 1
fi
echo "C++ 程序编译成功。"

# 定义测试用的 prompts 数组
declare -a PROMPTS=(
    "hello world"
    "  hello   world  "
    "你好，世界"
    "LLaMA.cpp is a great tool!"
    "What is the airspeed velocity of an unladen swallow?"
    "12345"
    "A B C D E"
    "    "
    "CLIP: Contrastive Language-Image Pre-Training"
    "複雑な日本語テキスト"
    " leading and trailing spaces "
    "an empty prompt"
)

#declare -a PROMPTS=(
    #"LLaMA.cpp is a great tool!"
    #"What is the airspeed velocity of an unladen swallow?"
    #"12345"
    #"你好，世界"
#)

# 循环处理每个 prompt
echo -e "\n开始批量处理 prompts..."
for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    # 通过环境变量传递唯一的后缀
    export OUTPUT_SUFFIX="text_${i}"

    echo -e "\n--- 处理 Prompt #${i} (Suffix: $OUTPUT_SUFFIX) ---"
    echo "Prompt: \"$PROMPT\""

    # 执行 C++ 程序
    #echo "正在运行 C++ embedding 程序..."
    #./build/bin/llama-tokenize -m /root/autodl-fs/bge-gguf/BGE-VL-large-text.gguf  -p "$PROMPT"  
    #exit
    ./build/bin/llama-embedding -m /root/autodl-fs/bge-gguf/BGE-VL-large-text.gguf  -p "$PROMPT"  --n-gpu-layers 99  -c 257  --pooling cls  -t 1
    if [ $? -ne 0 ]; then
        echo "C++ embedding 程序执行失败，跳过此 prompt。"
        continue
    fi

    # 执行 Python 程序
    echo "正在运行 Python embedding 程序..."
    python get_processor_result.py --prompt "$PROMPT" --model_path /root/autodl-tmp/Model/BGE-VL-large
    if [ $? -ne 0 ]; then
        echo "Python embedding 程序执行失败，跳过此 prompt。"
        continue
    fi
done

echo -e "\n--- 所有 Prompts 处理完毕 ---\n"

# 运行对比脚本
echo "正在运行对比脚本..."
python compare/compare1.py "$OUT_DIR"
echo "对比完成。" 
