export CUDA_VISIBLE_DEVICES=-1
cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build  --target llama-embedding -j 
#gdb --args ./build/bin/llama-embedding -m /root/autodl-tmp/Model/BGE-VL-large-GGUF/BGE-VL-large-vision-F16.gguf --n-gpu-layers 0 --image /root/BGE-VL-result/preprocessed_image.jpg -c 257
gdb --args ./build/bin/llama-embedding -m /root/autodl-tmp/Model/BGE-VL-large-GGUF/BGE-VL-large-vision-F16.gguf --n-gpu-layers 0 --image /root/BGE-VL-result/sample.png -c 257 -t 1 --pooling cls
