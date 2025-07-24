export OUT_DIR="/root/tmp/llama.cpp/compare/text_suite"
export OUTPUT_SUFFIX="test_image"
export CUDA_VISIBLE_DEVICES=-1
cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build  --target llama-embedding -j 
model=/root/autodl-fs/bge-gguf/BGE-VL-large-vision.gguf 
gdb --args ./build/bin/llama-embedding -m $model --n-gpu-layers 0 --image /root/BGE-VL-result/sample.png -c 257 
#gdb --args ./build/bin/llama-embedding -m $model --n-gpu-layers 0 --image /root/BGE-VL-result/preprocessed_image.jpg -c 1024 
