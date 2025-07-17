export OUTPUT_SUFFIX=0
export OUT_DIR=/root/tmp/llama.cpp/compare/text

export CUDA_VISIBLE_DEVICES=0
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build  --target llama-embedding -j 
#gdb --args ./build/bin/llama-embedding -m /root/autodl-tmp/Model/BGE-VL-large-GGUF/BGE-VL-large-vision-F16.gguf --n-gpu-layers 0 --image /root/BGE-VL-result/sample.png -c 257 -t 1 --pooling cls

#gdb --args ./build/bin/llama-embedding -m /root/autodl-tmp/Model/BGE-VL-large-GGUF/BGE-VL-large-vision-F16.gguf --n-gpu-layers 0 --image /root/BGE-VL-result/preprocessed_image.jpg -c 257 -t 1 --pooling cls

gdb --args ./build/bin/llama-embedding -m /root/autodl-fs/bge-gguf/BGE-VL-large-text.gguf --n-gpu-layers 99 -p "hello" -c 257  --pooling cls -t 1

#gdb --args ./build/bin/llama-simple -m /root/autodl-fs/Qwen2-7B-Instruct/ggml-model-Q4_K_M-bf16.gguf  --n-gpu-layers 99 -p "hello" -c 257 -t 1 -n 1

#gdb --arg ./build/bin/llama-simple -m /root/autodl-fs/t5/gguf_model/t5-v1_1-xxl-encoder-Q4_K_M.gguf --n-gpu-layers 99 -p "hello" -c 257 -t 1




