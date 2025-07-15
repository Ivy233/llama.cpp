export CUDA_VISIBLE_DEVICES=-1
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build  --target llama-embedding -j 

# Original commands (commented out)
#gdb --args ./build/bin/llama-embedding -m /root/autodl-tmp/Model/BGE-VL-large-GGUF/BGE-VL-large-vision-F16.gguf --n-gpu-layers 99 --image /root/BGE-VL-result/sample.png -c 257 -t 1 --pooling cls
#gdb --args ./build/bin/llama-embedding -m /root/autodl-tmp/Model/BGE-VL-large-GGUF/BGE-VL-large-vision-F16.gguf --n-gpu-layers 99 --image /root/BGE-VL-result/preprocessed_image.jpg -c 257 -t 1 --pooling cls
#gdb --args ./build/bin/llama-embedding -m /root/autodl-tmp/Model/BGE-VL-large-GGUF/BGE-VL-large-text-F16.gguf --n-gpu-layers 99 -p "hello" -c 257 -t 1 

# New commands with the new GGUF files
# Vision model with sample.png
#gdb --args ./build/bin/llama-embedding -m /root/autodl-fs/bge-gguf/BGE-VL-large-vision.gguf --n-gpu-layers 99 --image /root/BGE-VL-result/sample.png -c 257 -t 1 --pooling cls

# Vision model with preprocessed_image.jpg
#gdb --args ./build/bin/llama-embedding -m /root/autodl-fs/bge-gguf/BGE-VL-large-vision.gguf --n-gpu-layers 99 --image /root/BGE-VL-result/preprocessed_image.jpg -c 257 -t 1 --pooling cls

# Text model with "hello" prompt
gdb --args ./build/bin/llama-embedding -m /root/autodl-fs/bge-gguf/BGE-VL-large-text.gguf --n-gpu-layers 99 -p "hello" -c 257 -t 1 


