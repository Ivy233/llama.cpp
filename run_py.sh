#!/bin/bash


# python get_processor_result.py --mode image --input "/root/BGE-VL-result/sample.png"

#python get_processor_result.py --mode image --input "/root/BGE-VL-result/preprocessed_image.jpg"

python get_processor_result.py --mode text --input "  hello   world" --output_dir /root/tmp/llama.cpp/compare/text

# python get_processor_result.py --mode text --input "你好，这是一个测试"

# python get_processor_result.py --mode text --input "hello" --output_dir "/custom/output/path"
