set -e
set -x
clear
tgt_model=/root/autodl-fs/uniontech-yourong/yourong7B-Instruct-GGUF/yourong_7B_bf16_Q4_K_M.gguf



cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_CUDA_FA_ALL_QUANTS=0 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_NATIVE=ON
cmake --build build  --target llama-simple -j 50
temp=0.1
file=prompt.txt
only_temp=true
prompt="Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point."


predict=100
base_cmd="CUDA_VISIBLE_DEVICES=0 gdb --args ./build/bin/llama-simple \
-m ${tgt_model} -p \"${prompt}\" --predict ${predict} -c 2048 --temp ${temp} -ngl 0"

if [ "$only_temp" = true ]; then
    final_cmd="${base_cmd} --samplers temperature -t 1"
else
    final_cmd="${base_cmd} \
    --top-k 20 \
    --top-p 0.8 \
    --repeat-penalty 1.05 \
    --samplers 'top_k;top_p;temperature;penalties'"
fi

echo "final_cmd" : ${final_cmd}

eval $final_cmd
