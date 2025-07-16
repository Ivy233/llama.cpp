export OUT_DIR=/root/tmp/llama.cpp/compare/text
export CUDA_VISIBLE_DEVICES=-1
cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build  --target llama-embedding -j 
#gdb --args ./build/bin/llama-embedding -m /root/autodl-tmp/Model/BGE-VL-large-GGUF/BGE-VL-large-vision-F16.gguf --n-gpu-layers 0 --image /root/BGE-VL-result/sample.png -c 257 -t 1 --pooling cls

#gdb --args ./build/bin/llama-embedding -m /root/autodl-tmp/Model/BGE-VL-large-GGUF/BGE-VL-large-vision-F16.gguf --n-gpu-layers 0 --image /root/BGE-VL-result/preprocessed_image.jpg -c 257 -t 1 --pooling cls

gdb --args ./build/bin/llama-embedding -m /root/autodl-fs/bge-gguf/BGE-VL-large-text.gguf --n-gpu-layers 99 -p "hello world" -c 257  --pooling cls -t 1

#gdb --args ./build/bin/llama-simple -m /root/autodl-fs/Qwen2-7B-Instruct/ggml-model-Q4_K_M-bf16.gguf  --n-gpu-layers 99 -p "hello" -c 257 -t 1 -n 1

#gdb --arg ./build/bin/llama-simple -m /root/autodl-fs/t5/gguf_model/t5-v1_1-xxl-encoder-Q4_K_M.gguf --n-gpu-layers 99 -p "hello" -c 257 -t 1


======> CLIP处理: 添加</w>后的word: hello</w>
=======> vocab.get_ignore_merges(): 0
Before while loop, symbols:
enter while loop, offset: 0, word.size(): 9
enter while loop, offset: 1, word.size(): 9
enter while loop, offset: 2, word.size(): 9
enter while loop, offset: 3, word.size(): 9
enter while loop, offset: 4, word.size(): 9
enter while loop, offset: 5, word.size(): 9
=======> CLIP处理: 将</w>附加到前一个符号: 'o</w>'
push bigram to work_queue: he, left: 0, right: 1, size: 2, rank: 140
push bigram to work_queue: el, left: 1, right: 2, size: 2, rank: 33
push bigram to work_queue: ll, left: 2, right: 3, size: 2, rank: 1146
push bigram to work_queue: lo</w>, left: 3, right: 4, size: 6, rank: 2471
after add_new_bigram, symbols:
  [0] text='h' n=1 prev=-1 next=1
  [1] text='e' n=1 prev=0 next=2
  [2] text='l' n=1 prev=1 next=3
  [3] text='l' n=1 prev=2 next=4
  [4] text='o</w>' n=5 prev=3 next=-1
push bigram to work_queue: hel, left: 0, right: 1, size: 3, rank: 408
bigram.left: 1, left_symbol.next: 3
push bigram to work_queue: ell, left: 1, right: 3, size: 3, rank: 315
push bigram to work_queue: hell, left: 0, right: 1, size: 4, rank: 7899
bigram.left: 1, left_symbol.next: 4
push bigram to work_queue: ello</w>, left: 1, right: 4, size: 8, rank: 2001
push bigram to work_queue: hello</w>, left: 0, right: 1, size: 9, rank: 2795
bigram.left: 1, left_symbol.next: -1
bigram.left: 0, left_symbol.next: -1
after second while loop, symbols:
  [0] text='hello</w>' n=9 prev=-1 next=-1
  [1] text='' n=0 prev=0 next=-1
  [2] text='' n=0 prev=1 next=3
  [3] text='' n=0 prev=1 next=4
  [4] text='' n=0 prev=1 next=-1
=======> str: hello</w>
=======> token will be added to output: 3306
vocab.get_add_eos(): 1
49406 3306 49407

