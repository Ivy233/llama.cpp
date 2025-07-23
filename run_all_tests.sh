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
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build --target llama-embedding -j
cmake --build build --target llama-tokenize -j
if [ $? -ne 0 ]; then
    echo "C++ 程序编译失败，脚本终止。"
    exit 1
fi
echo "C++ 程序编译成功。"

# 定义测试用的 prompts 数组
declare -a PROMPTS=(
      # 基础测试
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

      # Unicode和特殊字符测试
      "Café naïve résumé"                    # 法语重音符号
      "Москва Санкт-Петербург"              # 俄语西里尔字母
      "한국어 테스트"                        # 韩语
      "ﺎﻠﻋﺮﺒﻳﺓ ﺎﺨﺘﺑﺍﺭ"                       # 阿拉伯语
      "🚀🌟💻🎯"                              # Emoji
      "α β γ δ ε ζ η θ"                     # 希腊字母
      "①②③④⑤"                               # 圈数字

      # 标点符号和特殊符号测试
      "Hello, world!"
      "What's that? It's amazing!"
      "Price: $100.50 (50% off)"
      "Email: test@example.com"
      "Path: /usr/bin/python3.9"
      "Math: 2+2=4, x²+y²=z²"
      "Quotes: \"Hello\" 'world'"

      # 混合语言测试
      "Hello 世界 Bonjour мир"
      "English中文日本語한국어"
      "Code: print('你好')"

      # 数字和字母数字组合
      "ABC123XYZ"
      "Version 1.2.3-beta"
      "ID: user123_test"
      "IPv4: 192.168.1.1"

      # 长度边界测试
      "a"                                    # 单字符
      "ab"                                   # 双字符
      "abcdefghijklmnopqrstuvwxyz"          # 长英文
      "你"                                   # 单个中文字符
      "这是一个相对较长的中文句子，用来测试tokenizer的处理能力"

      # 特殊空白字符测试
      "word1    word2"                         # Tab字符
      "line1\nline2"                        # 换行符
      "multiple   spaces   between"         # 多个空格

      # 大小写混合测试
      "MiXeD CaSe TeXt"
      "iPhone MacBook iOS"
      "HTML CSS JavaScript"

      # 缩写和特殊形式
      "don't won't can't shouldn't"
      "I'm you're they're we'll"
      "Dr. Prof. Mr. Mrs. vs. etc."

      # 技术术语
      "HTTP HTTPS REST API JSON XML"
      "machine learning AI transformer"
      "const fn = () => { return 42; }"

      # 内部数据集样例 - DSL查询语言
      "我想要几天后的文件，或者那些文件大小大于623KB或者小于953G，或者内容包含"5.数据安全保障"的文件。"
      "帮我找找文件名包含"resolution"且日期在元宵节之前的文件。还有啊，我也需要那种内容包含"创新无限"并且大小比"/financial_statement_2023.xlsx"小的文件。"

      # 内部数据集样例 - 函数调用
      "请获取英国邮编 WC2N 5DU 和 EC1A 1BB 的地址。另外，在法国查找一个名为"马赛"的市镇。"
      "获取亚马逊公司（AMZN）的股票统计数据以及美元兑印度卢比的汇率。"
  )

#declare -a PROMPTS=(
    #"LLaMA.cpp is a great tool!"
    #"What is the airspeed velocity of an unladen swallow?"
    #"12345"
    #"你好，世界"
    #"Москва Санкт-Петербург"
    #"line1\nline2"
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
    echo "正在运行 C++ embedding 程序..."
    #./build/bin/llama-tokenize -m /root/autodl-fs/bge-gguf/BGE-VL-large-text.gguf  -p "$PROMPT"  
    #exit
    ./build/bin/llama-embedding -m /root/autodl-fs/bge-gguf/BGE-VL-large-text.gguf  -p "$PROMPT"  --n-gpu-layers 99  -c 77  --pooling mean  -t 1
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
