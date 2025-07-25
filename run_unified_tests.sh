#!/bin/bash
set -e
set -x

# Integrated text and image testing script
# Supports two modes: text and image
# Usage:
# ./run_unified_tests.sh text [gpu|cpu]    # Run text tests
# ./run_unified_tests.sh image [gpu|cpu]   # Run image tests
# ./run_unified_tests.sh both [gpu|cpu]    # Run both tests
# ./run_unified_tests.sh [gpu|cpu]         # Default to both mode
# 
# Default uses GPU acceleration, can choose CPU mode

# Parse arguments
MODE=${1:-both}
DEVICE=${2:-gpu}

# If first argument is gpu or cpu, adjust parameters
if [[ "$MODE" == "gpu" || "$MODE" == "cpu" ]]; then
    DEVICE=$MODE
    MODE="both"
fi

# Display help information
if [[ "$MODE" == "--help" || "$MODE" == "-h" || "$MODE" == "help" ]]; then
    echo "BGE-VL Unified Testing Script"
    echo ""
    echo "Usage:"
    echo "  $0 [mode] [device]"
    echo ""
    echo "Mode options:"
    echo "  text  - Run text tests only"
    echo "  image - Run image tests only"
    echo "  both  - Run text and image tests (default)"
    echo ""
    echo "Device options:"
    echo "  gpu - Use GPU acceleration (default, n-gpu-layers=99)"
    echo "  cpu - Use CPU mode (n-gpu-layers=0)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Default: both mode, GPU acceleration"
    echo "  $0 gpu                # Both mode, GPU acceleration"
    echo "  $0 cpu                # Both mode, CPU mode"
    echo "  $0 text gpu           # Text tests, GPU acceleration"
    echo "  $0 image cpu          # Image tests, CPU mode"
    echo "  $0 both gpu           # Text+Image tests, GPU acceleration"
    exit 0
fi

# Interrupt handler function
cleanup() {
    echo -e "\n❌ Script interrupted!"
    echo "Interrupt time: $(date)"
    if [ -n "$CURRENT_TEST_INDEX" ]; then
        echo "Interrupt location: Processing ${TEST_TYPE} test #$CURRENT_TEST_INDEX"
        if [ -n "$CURRENT_TEST_ITEM" ]; then
            echo "Interrupted test content: \"$CURRENT_TEST_ITEM\""
        fi
        echo "Completed: $CURRENT_TEST_INDEX / ${#TEST_ITEMS[@]} test cases"
    else
        echo "Interrupt location: Script initialization phase"
    fi
    exit 1
}

# Register interrupt handler
trap cleanup SIGINT SIGTERM

# Validate parameters
if [[ "$MODE" != "text" && "$MODE" != "image" && "$MODE" != "both" ]]; then
    echo "Error: Invalid test mode '$MODE'"
    echo "Usage: $0 [text|image|both] [gpu|cpu]"
    exit 1
fi

if [[ "$DEVICE" != "gpu" && "$DEVICE" != "cpu" ]]; then
    echo "Error: Invalid device type '$DEVICE'"
    echo "Usage: $0 [text|image|both] [gpu|cpu]"
    exit 1
fi

# Set GPU layers
if [[ "$DEVICE" == "gpu" ]]; then
    GPU_LAYERS="--n-gpu-layers 99"
    echo "🚀 Using GPU acceleration mode (n-gpu-layers=99)"
else
    GPU_LAYERS="--n-gpu-layers 0"
    echo "🖥️  Using CPU mode (n-gpu-layers=0)"
fi

echo "=== BGE-VL Unified Testing Script ==="
echo "Test mode: $MODE"
echo "Device type: $DEVICE"
echo "Start time: $(date)"

# Compile C++ embedding program
echo "Compiling C++ embedding program..."
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build --target llama-embedding -j
if [ $? -ne 0 ]; then
    echo "C++ program compilation failed, script terminated."
    exit 1
fi
echo "C++ program compilation successful."

# Function: Run text tests
run_text_tests() {
    echo -e "\n🔤 Starting text tests..."
    
    # Set text test environment
    export OUT_DIR=/root/tmp/llama.cpp/compare/text_suite
    export TEST_TYPE="text"
    
    # Clean old test results
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
    echo "Text test output directory: $OUT_DIR"
    
    declare -a TEXT_PROMPTS=(
      # Basic tests
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

      # Unicode and special character tests
      "Café naïve résumé"                    # French accents
      "Москва Санкт-Петербург"              # Russian Cyrillic
      "한국어 테스트"                        # Korean
      "ﺎﻠﻋﺮﺒﻳﺓ ﺎﺨﺘﺑﺍﺭ"                       # Arabic
      "🚀🌟💻🎯"                              # Emoji
      "α β γ δ ε ζ η θ"                     # Greek letters
      "①②③④⑤"                               # Circled numbers

      # Punctuation and special symbol tests
      "Hello, world!"
      "What's that? It's amazing!"
      "Price: $100.50 (50% off)"
      "Email: test@example.com"
      "Path: /usr/bin/python3.9"
      "Math: 2+2=4, x²+y²=z²"
      "Quotes: \"Hello\" 'world'"

      # Mixed language tests
      "Hello 世界 Bonjour мир"
      "English中文日本語한국어"
      "Code: print('你好')"

      # Number and alphanumeric combinations
      "ABC123XYZ"
      "Version 1.2.3-beta"
      "ID: user123_test"
      "IPv4: 192.168.1.1"

      # Length boundary tests
      "a"                                    # Single character
      "ab"                                   # Double character
      "abcdefghijklmnopqrstuvwxyz"          # Long English
      "你"                                   # Single Chinese character
      "这是一个相对较长的中文句子，用来测试tokenizer的处理能力"

      # Special whitespace character tests
      "word1    word2"                         # Tab character
      "line1\nline2"                        # Newline character
      "multiple   spaces   between"         # Multiple spaces

      # Mixed case tests
      "MiXeD CaSe TeXt"
      "iPhone MacBook iOS"
      "HTML CSS JavaScript"

      # Abbreviations and special forms
      "don't won't can't shouldn't"
      "I'm you're they're we'll"
      "Dr. Prof. Mr. Mrs. vs. etc."

      # Technical terms
      "HTTP HTTPS REST API JSON XML"
      "machine learning AI transformer"
      "const fn = () => { return 42; }"

      #"帮我找找文件名包含"resolution"且日期在元宵节之前的文件。还有啊，我也需要那种内容包含"创新无限"并且大小比"/financial_statement_2023.xlsx"小的文件。"
      #"请获取英国邮编 WC2N 5DU 和 EC1A 1BB 的地址。另外，在法国查找一个名为"马赛"的市镇。"
  )
  
    # Define text test cases
    #declare -a TEXT_PROMPTS=(
        #"hello world"
        #"你好，世界"
        #"CLIP: Contrastive Language-Image Pre-Training"
        #"machine learning AI transformer"
        #"Café naïve résumé"
        #"🚀🌟💻🎯"
    #)
        
    # Loop through each text prompt
    for i in "${!TEXT_PROMPTS[@]}"; do
        PROMPT="${TEXT_PROMPTS[$i]}"
        export CURRENT_TEST_INDEX=$i
        export CURRENT_TEST_ITEM="$PROMPT"
        export OUTPUT_SUFFIX="text_${i}"

        echo -e "\n=== Text Test #${i} (Total ${#TEXT_PROMPTS[@]}) ==="
        echo "Current time: $(date)"
        echo "Output suffix: $OUTPUT_SUFFIX"
        echo -e "\n📝 Current text content:"
        echo "\"$PROMPT\""
        
        # Execute C++ program
        echo -e "\n🚀 Executing C++ embedding program..."
        ./build/bin/llama-embedding -m /root/autodl-fs/bge-gguf/BGE-VL-large-text.gguf -p "$PROMPT" $GPU_LAYERS 
        if [ $? -ne 0 ]; then
            echo "C++ embedding program execution failed, skipping this test."
            continue
        fi

        # Execute Python program
        echo -e "\n🐍 Executing Python embedding program..."
        python get_processor_result.py --prompt "$PROMPT" --model_path /root/autodl-tmp/Model/BGE-VL-large
        if [ $? -ne 0 ]; then
            echo "Python embedding program execution failed, skipping this test."
            continue
        fi
        
        echo "✅ Text test #${i} completed"
    done
    
    echo -e "\n🔤 Text test batch completed"
    
    # Run text comparison analysis
    echo "Running text comparison analysis..."
    python compare/compare1.py "$OUT_DIR"
    echo "Text comparison analysis completed."
}

# Function: Run image tests  
run_image_tests() {
    echo -e "\n🖼️  Starting image tests..."
    
    # Set image test environment
    export OUT_DIR=/root/tmp/llama.cpp/compare/img_suite
    export TEST_TYPE="image"
    
    # Clean old test results
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
    echo "Image test output directory: $OUT_DIR"
    
    # Define image test cases using test_images directory
    declare -a IMAGE_PATHS=()
    
    # Use test_images directory with different sizes and formats
    TEST_IMG_DIR="/root/tmp/llama.cpp/test_images"
    if [ -d "$TEST_IMG_DIR" ]; then
        # Add images of different sizes for comprehensive testing
        for img in "$TEST_IMG_DIR"/*.{png,jpg,jpeg}; do
            if [ -f "$img" ] && [ -s "$img" ]; then  # Check file exists and is not empty
                IMAGE_PATHS+=("$img")
            fi
        done
    fi
    
    # Fallback to original test images if test_images directory is empty
    if [ ${#IMAGE_PATHS[@]} -eq 0 ]; then
        if [ -f "/root/BGE-VL-result/sample.png" ]; then
            IMAGE_PATHS+=("/root/BGE-VL-result/sample.png")
        fi
        if [ -f "/root/BGE-VL-result/preprocessed_image.jpg" ]; then
            IMAGE_PATHS+=("/root/BGE-VL-result/preprocessed_image.jpg")
        fi
        if [ -f "/root/tmp/llama.cpp/tools/mtmd/test-1.jpeg" ]; then
            IMAGE_PATHS+=("/root/tmp/llama.cpp/tools/mtmd/test-1.jpeg")
        fi
    fi
    
    if [ ${#IMAGE_PATHS[@]} -eq 0 ]; then
        echo "❌ No test images found, skipping image tests"
        return 1
    fi
    
    echo "Found ${#IMAGE_PATHS[@]} test images with different sizes and formats"
    
    # Loop through each image
    for i in "${!IMAGE_PATHS[@]}"; do
        IMAGE_PATH="${IMAGE_PATHS[$i]}"
        export CURRENT_TEST_INDEX=$i
        export CURRENT_TEST_ITEM="$IMAGE_PATH"
        
        # Get image file extension as output suffix
        IMAGE_BASENAME=$(basename "$IMAGE_PATH")
        IMAGE_EXT="${IMAGE_BASENAME##*.}"
        export OUTPUT_SUFFIX="${IMAGE_EXT}_${i}"

        echo -e "\n=== Image Test #${i} (Total ${#IMAGE_PATHS[@]}) ==="
        echo "Current time: $(date)"
        echo "Output suffix: $OUTPUT_SUFFIX"
        echo -e "\n🖼️  Current image file:"
        echo "\"$IMAGE_PATH\""
        
        # Check if image file exists
        if [ ! -f "$IMAGE_PATH" ]; then
            echo "❌ Image file does not exist, skipping this test."
            continue
        fi
        
        # Execute C++ program
        echo -e "\n🚀 Executing C++ embedding program..."
        ./build/bin/llama-embedding -m /root/autodl-fs/bge-gguf/BGE-VL-large-vision.gguf --image "$IMAGE_PATH" $GPU_LAYERS -c 257
        if [ $? -ne 0 ]; then
            echo "C++ embedding program execution failed, skipping this test."
            continue
        fi

        # Execute Python program
        echo -e "\n🐍 Executing Python embedding program..."
        python get_processor_result.py --image_path "$IMAGE_PATH" --model_path /root/autodl-tmp/Model/BGE-VL-large
        if [ $? -ne 0 ]; then
            echo "Python embedding program execution failed, skipping this test."
            continue
        fi
        
        echo "✅ Image test #${i} completed"
    done
    
    echo -e "\n🖼️  Image test batch completed"
    
    # Run image comparison analysis
    echo "Running image comparison analysis..."
    python compare/compare1.py "$OUT_DIR"
    echo "Image comparison analysis completed."
}

# Function: Display comprehensive results summary
show_summary() {
    echo -e "\n📊 === Test Results Summary ==="
    echo "Test completion time: $(date)"
    
    # Statistics variables
    local TEXT_COUNT=0
    local IMAGE_COUNT=0
    local TOTAL_COUNT=0
    
    if [[ "$MODE" == "text" || "$MODE" == "both" ]]; then
        echo -e "\n🔤 Text test results:"
        if [ -d "/root/tmp/llama.cpp/compare/text_suite" ]; then
            TEXT_FILES=$(find /root/tmp/llama.cpp/compare/text_suite -name "*.txt" | wc -l)
            TEXT_COUNT=$((TEXT_FILES / 2))  # Each test generates cpp+py two files
            echo "  - Generated embedding files count: $TEXT_FILES"
            echo "  - Completed test cases count: $TEXT_COUNT"
            echo "  - Results directory: /root/tmp/llama.cpp/compare/text_suite"
            ls -la /root/tmp/llama.cpp/compare/text_suite/ | head -10
        else
            echo "  - Text test results not found"
        fi
    fi
    
    if [[ "$MODE" == "image" || "$MODE" == "both" ]]; then
        echo -e "\n🖼️  Image test results:"
        if [ -d "/root/tmp/llama.cpp/compare/img_suite" ]; then
            IMG_FILES=$(find /root/tmp/llama.cpp/compare/img_suite -name "*.txt" | wc -l)
            IMAGE_COUNT=$((IMG_FILES / 2))  # Each test generates cpp+py two files
            echo "  - Generated embedding files count: $IMG_FILES"
            echo "  - Completed test cases count: $IMAGE_COUNT"
            echo "  - Results directory: /root/tmp/llama.cpp/compare/img_suite"
            ls -la /root/tmp/llama.cpp/compare/img_suite/ | head -10
        else
            echo "  - Image test results not found"
        fi
    fi
    
    # Display overall statistics
    TOTAL_COUNT=$((TEXT_COUNT + IMAGE_COUNT))
    if [ $TOTAL_COUNT -gt 0 ]; then
        echo -e "\n📈 === Overall Statistics ==="
        echo "✅ Text test cases: $TEXT_COUNT"
        echo "✅ Image test cases: $IMAGE_COUNT"  
        echo "✅ Total test cases: $TOTAL_COUNT"
        echo ""
        echo "🔍 Similarity analysis results:"
        if [ -f "/root/tmp/llama.cpp/compare/compare1.py" ]; then
            echo "Run the following command to view detailed similarity report:"
            echo "  cd /root/tmp/llama.cpp/compare && python compare1.py"
        fi
    fi
    
    echo -e "\n✅ All tests completed!"
}

# Main program execution logic
case "$MODE" in
    "text")
        run_text_tests
        ;;
    "image")
        run_image_tests
        ;;
    "both")
        run_text_tests
        run_image_tests
        ;;
esac

# Display results summary
show_summary

# Auto-run similarity analysis (if test results exist)
if [[ "$MODE" == "both" ]] || [[ -d "/root/tmp/llama.cpp/compare/text_suite" && -d "/root/tmp/llama.cpp/compare/img_suite" ]]; then
    echo -e "\n🔍 === Auto-run Similarity Analysis ==="
    echo "Analyzing similarity of all test results..."
    cd /root/tmp/llama.cpp/compare && python compare1.py
    echo -e "\n📊 Similarity analysis completed!"
fi

echo -e "\n🎉 BGE-VL Unified Testing Script Execution Completed"
