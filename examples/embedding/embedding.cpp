#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "clip.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <ctime>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib> // For std::getenv

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// Global debug control variable
static bool g_enable_debug = false;

// Debug output macro
#define DEBUG_PRINTF(...) do { if (g_enable_debug) printf(__VA_ARGS__); } while(0)

static std::vector<std::string> split_lines(const std::string & s, const std::string & separator = "\n") {
    std::vector<std::string> lines;
    size_t start = 0;
    size_t end = s.find(separator);

    while (end != std::string::npos) {
        lines.push_back(s.substr(start, end - start));
        start = end + separator.length();
        end = s.find(separator, start);
    }

    lines.push_back(s.substr(start)); // Add the last part

    return lines;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        // For sequence-level pooling, only set logits=true for the last token
        bool need_logits = (i == n_tokens - 1);
        common_batch_add(batch, tokens[i], i, { seq_id }, need_logits);
    }
}

static void batch_encode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm, bool is_image = false) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_self_clear(ctx);

    DEBUG_PRINTF("=== BGE-VL Neural Network Inference Start ===\n");
    DEBUG_PRINTF("Input batch.n_tokens: %d\n", batch.n_tokens);
    DEBUG_PRINTF("Input embeddings dimension: %d\n", n_embd);
    DEBUG_PRINTF("Context size: %d\n", llama_n_ctx(ctx));
    if (batch.embd != nullptr) {
        DEBUG_PRINTF("Input embeds (first 10): ");
        for (int i = 0; i < 10 && i < n_embd; ++i) {
            DEBUG_PRINTF("%.6f ", batch.embd[i]);
        }
        DEBUG_PRINTF("\n");
    }

    // run model
    if (llama_encode(ctx, batch) < 0) {
        LOG_ERR("%s: failed to process\n", __func__);
        return;
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int embd_pos = 0;

        // Extract embeddings based on context's pooling type
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            embd = llama_get_embeddings_seq(ctx, 0);
            embd_pos = 0;
            GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }

        DEBUG_PRINTF("BGE-VL raw output (first 10): ");
        for (int j = 0; j < 10 && j < n_embd; ++j) {
            DEBUG_PRINTF("%.6f ", embd[j]);
        }
        DEBUG_PRINTF("\n");

        float * out = output + embd_pos * n_embd;
        common_embd_normalize(embd, out, n_embd, embd_norm);

        DEBUG_PRINTF("BGE-VL final embedding (first 10): ");
        for (int j = 0; j < 10 && j < n_embd; ++j) {
            DEBUG_PRINTF("%.6f ", out[j]);
        }
        DEBUG_PRINTF("\n");
        
        // For sequence-level pooling, we only need one embedding per sequence
        if (pooling_type != LLAMA_POOLING_TYPE_NONE) {
            break;
        }
    }
}

// BICUBIC插值核函数 (匹配PIL的BICUBIC实现)
static float bicubic_kernel(float x) {
    x = std::abs(x);
    if (x <= 1.0f) {
        return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    } else if (x < 2.0f) {
        return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    } else {
        return 0.0f;
    }
}

// BICUBIC interpolation function (4x4 sampling, matches Python PIL.Image.BICUBIC)
static float bicubic_interpolate(const std::vector<float>& temp, int longer_side, float sx, float sy, int c) {
    const int x0 = static_cast<int>(std::floor(sx));
    const int y0 = static_cast<int>(std::floor(sy));
    
    const float dx = sx - x0;
    const float dy = sy - y0;
    
    float result = 0.0f;
    
    // Use 4x4 grid for BICUBIC interpolation
    for (int j = -1; j <= 2; j++) {
        for (int i = -1; i <= 2; i++) {
            int px = std::max(0, std::min(longer_side - 1, x0 + i));
            int py = std::max(0, std::min(longer_side - 1, y0 + j));
            
            const int idx = 3 * (py * longer_side + px) + c;
            
            if (idx >= 0 && idx < static_cast<int>(temp.size())) {
                float weight_x = bicubic_kernel(dx - i);
                float weight_y = bicubic_kernel(dy - j);
                result += temp[idx] * weight_x * weight_y;
            }
        }
    }
    
    return result;
}

llama_batch llama_image_preprocess(const uint8_t* image_data, int width, int height, int channels, int target_size, int patch_size)
{
    llama_batch batch = {};

    // Enhanced input validation
    if (!image_data) {
        LOG_ERR("%s: image_data is NULL\n", __func__);
        return batch;
    }
    if (width <= 0 || height <= 0) {
        LOG_ERR("%s: Invalid image dimensions: %dx%d\n", __func__, width, height);
        return batch;
    }
    if (channels != 3) {
        LOG_ERR("%s: Expected 3 channels, got %d\n", __func__, channels);
        return batch;
    }
    if (target_size <= 0 || patch_size <= 0) {
        LOG_ERR("%s: Invalid target_size (%d) or patch_size (%d)\n", __func__, target_size, patch_size);
        return batch;
    }
    if (target_size % patch_size != 0) {
        LOG_ERR("%s: target_size (%d) must be divisible by patch_size (%d)\n", __func__, target_size, patch_size);
        return batch;
    }

    const int longer_side = std::max(width, height);
    const float scale = std::min(
        static_cast<float>(target_size) / width,
        static_cast<float>(target_size) / height
    );
    const uint8_t bc[3] = {122, 116, 104}; // background color in RGB from LLaVA
    
    // Prevent buffer overflow - check buffer sizes
    const size_t input_pixel_count = static_cast<size_t>(width) * height;
    const size_t temp_pixel_count = static_cast<size_t>(longer_side) * longer_side;
    const size_t processed_pixel_count = static_cast<size_t>(target_size) * target_size;
    
    // Prevent integer overflow and excessive memory allocation
    if (temp_pixel_count > SIZE_MAX / 3 || processed_pixel_count > SIZE_MAX / 3) {
        LOG_ERR("%s: Image dimensions too large for safe memory allocation\n", __func__);
        return batch;
    }
    
    const size_t temp_size = temp_pixel_count * 3;
    const size_t processed_size = processed_pixel_count * 3;
    
    // Limit maximum memory usage (100MB)
    if (temp_size > 100000000 || processed_size > 100000000) {
        LOG_ERR("%s: Image requires too much memory: temp=%zu, processed=%zu\n", __func__, temp_size, processed_size);
        return batch;
    }
    
    std::vector<float> processed(processed_size);
    std::vector<float> temp(temp_size);

    if (width != height) {
        // fill with background color
        for (size_t i = 0; i < temp.size(); i++) {
            temp[i] = bc[i % 3];
        }

        // copy from the input image with bounds checking
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                const size_t src_idx = static_cast<size_t>(y) * width + x;
                const size_t dst_idx = static_cast<size_t>(y) * longer_side + x;
                
                if (src_idx >= input_pixel_count || dst_idx >= temp_pixel_count) {
                    LOG_ERR("%s: Index out of bounds in image copy: src=%zu, dst=%zu\n", __func__, src_idx, dst_idx);
                    return batch;
                }
                
                for (int c = 0; c < channels; c++) {
                    const size_t src_offset = src_idx * 3 + c;
                    const size_t dst_offset = dst_idx * 3 + c;
                    temp[dst_offset] = image_data[src_offset];
                }
            }
        }
    } else {
        // Direct copy with bounds checking
        const size_t copy_size = static_cast<size_t>(width) * height * channels;
        if (copy_size > temp.size()) {
            LOG_ERR("%s: Copy size (%zu) exceeds temp buffer size (%zu)\n", __func__, copy_size, temp.size());
            return batch;
        }
        for(size_t i = 0; i < copy_size; i++){
            temp[i] = image_data[i];
        }
    }

    const int nx3 = int(longer_side * scale + 0.5f);
    const int ny3 = int(longer_side * scale + 0.5f);
    const float m3[] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float s3[] = {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                const float sx = (x + 0.5f) / scale - 0.5f;
                const float sy = (y + 0.5f) / scale - 0.5f;

                // Use BICUBIC interpolation (matches Python PIL BICUBIC)
                const float v = bicubic_interpolate(temp, longer_side, sx, sy, c);
                
                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                // CHW format: BGE-VL expects channel separation
                const int i = c * (nx3 * ny3) + y * nx3 + x;
                processed[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }

    // Calculate patch count using target_size
    int num_patches_per_dim = target_size / patch_size;
    int num_patches = num_patches_per_dim * num_patches_per_dim;
    
    // Check calculation results
    if (num_patches <= 0 || num_patches > 10000) {
        LOG_ERR("%s: Invalid number of patches: %d\n", __func__, num_patches);
        return batch;
    }
    
    const int embd_size = target_size * target_size * 3;
    if (embd_size <= 0) {
        LOG_ERR("%s: Invalid embedding size: %d\n", __func__, embd_size);
        return batch;
    }
    
    batch = llama_batch_init(num_patches, embd_size, 1);
    
    // Complete batch initialization check
    if (batch.embd == nullptr) {
        LOG_ERR("%s: Failed to allocate batch embeddings\n", __func__);
        return batch;
    }
    if (batch.seq_id == nullptr) {
        LOG_ERR("%s: Failed to allocate batch seq_id\n", __func__);
        llama_batch_free(batch);
        batch = {};
        return batch;
    }
    if (batch.pos == nullptr) {
        LOG_ERR("%s: Failed to allocate batch pos\n", __func__);
        llama_batch_free(batch);
        batch = {};
        return batch;
    }
    if (batch.n_seq_id == nullptr) {
        LOG_ERR("%s: Failed to allocate batch n_seq_id\n", __func__);
        llama_batch_free(batch);
        batch = {};
        return batch;
    }

    batch.n_tokens = num_patches;
    
    // Boundary check: ensure data size matches
    if (processed.size() != static_cast<size_t>(embd_size)) {
        LOG_ERR("%s: Processed array size (%zu) doesn't match expected size (%d)\n", 
                __func__, processed.size(), embd_size);
        llama_batch_free(batch);
        batch = {};
        return batch;
    }
    
    // Safe data copy
    for (int i = 0; i < embd_size; ++i) {
        batch.embd[i] = processed[i];
    }
    
    // Initialize batch metadata with bounds checking
    for (int i = 0; i < num_patches; i++) {
        if (i >= batch.n_tokens || batch.seq_id[i] == nullptr) {
            LOG_ERR("%s: Invalid batch index %d (n_tokens=%d)\n", __func__, i, batch.n_tokens);
            llama_batch_free(batch);
            batch = {};
            return batch;
        }
        batch.seq_id[i][0] = 0;  
        batch.n_seq_id[i] = 1;  
        batch.pos[i] = i;      
    }
    
    DEBUG_PRINTF("=== BGE-VL Image Preprocessing Debug Info ===\n");
    DEBUG_PRINTF("Input image size: %dx%d, channels: %d\n", width, height, channels);
    DEBUG_PRINTF("Target size: %d, scale: %.6f\n", target_size, scale);
    DEBUG_PRINTF("Normalization means: [%.6f, %.6f, %.6f]\n", m3[0], m3[1], m3[2]);
    DEBUG_PRINTF("Normalization stds: [%.6f, %.6f, %.6f]\n", s3[0], s3[1], s3[2]);
    
    DEBUG_PRINTF("Raw pixel values (first 10): ");
    for (int i = 0; i < 10 && i < width * height * channels; ++i) {
        DEBUG_PRINTF("%d ", image_data[i]);
    }
    DEBUG_PRINTF("\n");
    
    DEBUG_PRINTF("Processed pixel values (first 10): ");
    for (int i = 0; i < 10 && i < target_size * target_size * 3; ++i) {
        DEBUG_PRINTF("%.10f ", processed[i]);
    }
    DEBUG_PRINTF("\n");

    return batch;
}

// Function to process image and get embeddings
static bool process_image_embedding(llama_context * ctx, const std::string & image_path, float * output, int n_embd, int embd_norm) {
    // Parameter validation
    if (!ctx || image_path.empty() || !output || n_embd <= 0) {
        LOG_ERR("%s: Invalid input parameters\n", __func__);
        return false;
    }
    
    int width = 0, height = 0, channels = 0;

    unsigned char * rgb_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!rgb_data) {
        const char* error_reason = stbi_failure_reason();
        LOG_ERR("%s: Failed to load image file %s: %s\n", __func__, image_path.c_str(), error_reason ? error_reason : "unknown error");
        return false;
    }
    
    // Use RAII for resource management
    struct ImageDataDeleter {
        unsigned char* data;
        ImageDataDeleter(unsigned char* d) : data(d) {}
        ~ImageDataDeleter() { if (data) stbi_image_free(data); }
    } image_guard(rgb_data);
    
    // Additional sanity checks
    if (width <= 0 || height <= 0 || width > 10000 || height > 10000) {
        LOG_ERR("%s: Invalid image dimensions: %dx%d\n", __func__, width, height);
        return false;
    }

    auto model = llama_get_model(ctx);
    
    if (!model) {
        LOG_ERR("%s: Failed to get model from context\n", __func__);
        return false;
    }
    
    auto patch_size = get_n_image_patch_size(ctx);
    
    if (patch_size <= 0 || patch_size > 256) {
        LOG_ERR("%s: Invalid patch_size: %d\n", __func__, patch_size);
        return false;
    }
    
    struct llama_batch llm_batch = llama_image_preprocess(rgb_data, width, height, channels, 224, patch_size);
    
    if (llm_batch.embd == nullptr || llm_batch.n_tokens <= 0) {
        LOG_ERR("%s: Image preprocessing failed\n", __func__);
        return false;
    }
    
    batch_encode(ctx, llm_batch, output, 1, n_embd, embd_norm, true);

    return true;
}

std::string get_file_extension(const std::string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos != std::string::npos) {
        return filename.substr(pos + 1);
    }
    return "";
}

void save_embedding_to_file(const float * emb, int n_embd, const std::string & type, const std::string & fname_prefix) {
    // Parameter validation
    if (!emb || n_embd <= 0) {
        LOG_ERR("%s: Invalid parameters: emb=%p, n_embd=%d\n", __func__, static_cast<const void*>(emb), n_embd);
        return;
    }
    
    if (type.empty() || fname_prefix.empty()) {
        LOG_ERR("%s: Empty type or fname_prefix\n", __func__);
        return;
    }
    
    // Prevent oversized embeddings
    if (n_embd > 100000) {
        LOG_ERR("%s: Embedding size too large: %d\n", __func__, n_embd);
        return;
    }
    
    const char* suffix_env = std::getenv("OUTPUT_SUFFIX");
    std::string suffix = suffix_env ? std::string(suffix_env) : "";

    const char* out_dir_env = std::getenv("OUT_DIR");
    std::string out_dir = out_dir_env ? std::string(out_dir_env) : "./compare";

    // Construct the full path with validation
    std::string output_filename;
    try {
        if (type == "img") {
            std::string ext = get_file_extension(fname_prefix);
            if (ext.empty()) {
                ext = "unknown";
            }
            output_filename = out_dir + "/cpp_" + (suffix.empty() ? ext : suffix) + "_embd.txt";
        } else if (type == "text") {
            output_filename = out_dir + "/cpp_" + (suffix.empty() ? "text" : suffix) + "_embd.txt";
        } else {
            LOG_ERR("%s: Unknown type: %s\n", __func__, type.c_str());
            return;
        }
        
        // Check filename length
        if (output_filename.length() > 1000) {
            LOG_ERR("%s: Output filename too long: %zu characters\n", __func__, output_filename.length());
            return;
        }
    } catch (const std::exception& e) {
        LOG_ERR("%s: Error constructing filename: %s\n", __func__, e.what());
        return;
    }

    // Open file and save with error handling
    std::ofstream out_file(output_filename);
    if (!out_file.is_open()) {
        LOG_ERR("%s: Failed to open file %s for writing\n", __func__, output_filename.c_str());
        return;
    }
    
    try {
        out_file << std::fixed << std::setprecision(6);
        for (int i = 0; i < n_embd; ++i) {
            // Check numeric validity
            if (!std::isfinite(emb[i])) {
                LOG_ERR("%s: Invalid embedding value at index %d: %f\n", __func__, i, emb[i]);
                out_file.close();
                return;
            }
            out_file << emb[i] << (i == n_embd - 1 ? "" : " ");
        }
        out_file << std::endl;
        
        if (!out_file.good()) {
            LOG_ERR("%s: Error writing to file %s\n", __func__, output_filename.c_str());
            return;
        }
        
        out_file.close();
    } catch (const std::exception& e) {
        LOG_ERR("%s: Exception while writing file: %s\n", __func__, e.what());
        out_file.close();
    }
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }

    // Check debug mode environment variable
    const char* debug_env = std::getenv("BGE_DEBUG");
    g_enable_debug = (debug_env != nullptr && std::string(debug_env) == "1");

    const char* suffix_env = std::getenv("OUTPUT_SUFFIX");
    std::string output_suffix = (suffix_env != nullptr) ? suffix_env : "";

    common_init();

    params.embedding = true;

    // utilize the full context
    if (params.n_batch < params.n_ctx) {
        LOG_WRN("%s: setting batch size to %d\n", __func__, params.n_ctx);
        params.n_batch = params.n_ctx;
    }

    // For non-causal models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;

    llama_backend_init();
    llama_numa_init(params.numa);

    // BGE-VL pooling strategy: based on actual test results
    // According to commit 43fdaecb tests, text model requires CLS pooling
    bool is_image = !params.image.empty();
    if (is_image) {
        // Image input: force CLS pooling (position 0), matches Python implementation
        params.pooling_type = LLAMA_POOLING_TYPE_CLS;
    } else {
        // Text input: based on tests, commit 43fdaecb text model requires CLS pooling
        params.pooling_type = LLAMA_POOLING_TYPE_CLS;
    }

    // load the model
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        LOG_ERR("%s: computing embeddings in encoder-decoder models is not supported\n", __func__);
        return 1;
    }

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }
    
    int n_embd_count = 0;

    // Allocate output for embeddings
    const int n_embd = llama_model_n_embd(model);
    std::vector<float> embeddings;
    float * emb = nullptr;
    if (is_image) {
        // Process image
        embeddings.resize(n_embd, 0);
        emb = embeddings.data();
        if (!process_image_embedding(ctx, params.image[0], emb, n_embd, params.embd_normalize)) {
            LOG_ERR("%s: failed to process image embedding\n", __func__);
            llama_backend_free();
            return 1;
        }
    } else {
        // split the prompt into lines
        std::vector<std::string> prompts = split_lines(params.prompt, params.embd_sep);

        // max batch size
        const uint64_t n_batch = params.n_batch;

        // tokenize the prompts and trim
        std::vector<std::vector<int32_t>> inputs;
        for (const auto & prompt : prompts) {
            auto inp = common_tokenize(ctx, prompt, true, true);
            
            // BGE-VL model limit: check if exceeds maximum context length
            const int max_ctx_length = llama_model_n_ctx_train(llama_get_model(ctx));
            if (inp.size() > max_ctx_length) {
                LOG_ERR("%s: prompt too long (%lld tokens), BGE-VL model only supports up to %d tokens\n", 
                        __func__, (long long int) inp.size(), max_ctx_length);
                LOG_ERR("%s: please shorten your prompt or split it into smaller parts\n", __func__);
                return 1;
            }
            
            if (inp.size() > n_batch) {
                LOG_ERR("%s: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                        __func__, (long long int) inp.size(), (long long int) n_batch);
                return 1;
            }
            inputs.push_back(inp);
        }
        
        // check if the last token is SEP
        for (auto & inp : inputs) {
            if (inp.empty() || inp.back() != llama_vocab_sep(vocab)) {
                LOG_WRN("%s: last token in the prompt is not SEP\n", __func__);
                LOG_WRN("%s: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header\n", __func__);
            }
        }

        // tokenization stats
        if (params.verbose_prompt) {
            for (int i = 0; i < (int) inputs.size(); i++) {
                LOG_INF("%s: prompt %d: '%s'\n", __func__, i, prompts[i].c_str());
                LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, inputs[i].size());
                for (int j = 0; j < (int) inputs[i].size(); j++) {
                    LOG("%6d -> '%s'\n", inputs[i][j], common_token_to_piece(ctx, inputs[i][j]).c_str());
                }
                LOG("\n\n");
            }
        }

        // initialize batch
        const int n_prompts = prompts.size();
        struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

        // count number of embeddings
        // BGE-VL: text always uses LAST pooling (sequence level), so always n_prompts embeddings
        n_embd_count = n_prompts;

        // allocate output
        embeddings.resize(n_embd_count * n_embd, 0);
        emb = embeddings.data();

        // break into batches
        int e = 0; // number of embeddings already stored
        int s = 0; // number of prompts in current batch
        for (int k = 0; k < n_prompts; k++) {
            // clamp to n_batch tokens
            auto & inp = inputs[k];

            const uint64_t n_toks = inp.size();

            // encode if at capacity
            if (batch.n_tokens + n_toks > n_batch) {
                float * out = emb + e * n_embd;
                batch_encode(ctx, batch, out, s, n_embd, params.embd_normalize, false);
                e += s;
                s = 0;
                common_batch_clear(batch);
            }

            // add to batch
            batch_add_seq(batch, inp, 0);
            s += 1;
        }

        // final batch
        float * out = emb + e * n_embd;
        batch_encode(ctx, batch, out, s, n_embd, params.embd_normalize, false);
        llama_batch_free(batch);
    }

    // Output embeddings
    if (params.embd_out.empty()) {
        LOG("\n");

        if (is_image) {
            LOG("image embedding: ");
            for (int i = 0; i < n_embd; i++) {
                if (params.embd_normalize == 0) {
                    LOG("%6.0f ", emb[i]);
                } else {
                    LOG("%9.6f ", emb[i]);
                }
            }
            
            save_embedding_to_file(emb, n_embd, "img", params.image[0]);

            LOG("\n");
        } else {
            save_embedding_to_file(emb, n_embd, "text", output_suffix);
            // print the first part of the embeddings or for a single prompt, the full embedding
            int n_prompts = is_image ? 1 : n_embd_count;
            for (int j = 0; j < n_prompts; j++) {
                LOG("embedding %d: ", j);
                for (int i = 0; i < (n_prompts > 1 ? std::min(16, n_embd) : n_embd); i++) {
                    if (params.embd_normalize == 0) {
                        LOG("%6.0f ", emb[j * n_embd + i]);
                    } else {
                        LOG("%9.6f ", emb[j * n_embd + i]);
                    }
                }
                LOG("\n");
            }
        }
    }

    if (params.embd_out == "json" || params.embd_out == "json+" || params.embd_out == "array") {
        const bool notArray = params.embd_out != "array";

        LOG(notArray ? "{\n  \"object\": \"list\",\n  \"data\": [\n" : "[");
        for (int j = 0;;) { // at least one iteration (one prompt)
            if (notArray) LOG("    {\n      \"object\": \"embedding\",\n      \"index\": %d,\n      \"embedding\": ",j);
            LOG("[");
            for (int i = 0;;) { // at least one iteration (n_embd > 0)
                LOG(params.embd_normalize == 0 ? "%1.0f" : "%1.7f", emb[j * n_embd + i]);
                i++;
                if (i < n_embd) LOG(","); else break;
            }
            LOG(notArray ? "]\n    }" : "]");
            j++;
            if (j < n_embd_count) LOG(notArray ? ",\n" : ","); else break;
        }
        LOG(notArray ? "\n  ]" : "]\n");

        if (params.embd_out == "json+" && n_embd_count > 1) {
            LOG(",\n  \"cosineSimilarity\": [\n");
            for (int i = 0;;) { // at least two iteration (n_embd_count > 1)
                LOG("    [");
                for (int j = 0;;) { // at least two iteration (n_embd_count > 1)
                    float sim = common_embd_similarity_cos(emb + i * n_embd, emb + j * n_embd, n_embd);
                    LOG("%6.2f", sim);
                    j++;
                    if (j < n_embd_count) LOG(", "); else break;
                }
                LOG(" ]");
                i++;
                if (i < n_embd_count) LOG(",\n"); else break;
            }
            LOG("\n  ]");
        }

        if (notArray) LOG("\n}\n");
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    // clean up
    llama_backend_free();

    return 0;
}
