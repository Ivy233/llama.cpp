cd /root/tmp/llama.cpp/test_images

# 224x224 (BGE-VL标准尺寸) - 3种格式
curl -L -o "std_224x224.png" "https://via.placeholder.com/224x224.png"
curl -L -o "std_224x224.jpg" "https://via.placeholder.com/224x224.jpg"
curl -L -o "std_224x224.jpeg" "https://via.placeholder.com/224x224/0000FF/FFFFFF.jpeg"

# 256x256 - 3种格式
curl -L -o "small_256x256.png" "https://via.placeholder.com/256x256.png"
curl -L -o "small_256x256.jpg" "https://via.placeholder.com/256x256.jpg"
curl -L -o "small_256x256.jpeg" "https://via.placeholder.com/256x256/FF0000/FFFFFF.jpeg"

# 512x512 - 3种格式  
curl -L -o "medium_512x512.png" "https://via.placeholder.com/512x512.png"
curl -L -o "medium_512x512.jpg" "https://via.placeholder.com/512x512.jpg"
curl -L -o "medium_512x512.jpeg" "https://via.placeholder.com/512x512/00FF00/000000.jpeg"

# 640x480 - 3种格式
curl -L -o "rect_640x480.png" "https://via.placeholder.com/640x480.png"
curl -L -o "rect_640x480.jpg" "https://via.placeholder.com/640x480.jpg"
curl -L -o "rect_640x480.jpeg" "https://via.placeholder.com/640x480/FFFF00/000000.jpeg"

# 1024x768 - 3种格式
curl -L -o "large_1024x768.png" "https://via.placeholder.com/1024x768.png"
curl -L -o "large_1024x768.jpg" "https://via.placeholder.com/1024x768.jpg"
curl -L -o "large_1024x768.jpeg" "https://via.placeholder.com/1024x768/FF00FF/FFFFFF.jpeg"

echo "所有测试图片下载完成！"
ls -la *.png *.jpg *.jpeg | wc -l
