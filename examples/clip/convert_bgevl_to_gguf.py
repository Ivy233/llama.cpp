import json
import torch
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoTokenizer

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent.parent / "gguf-py"))
import gguf

CLIP_TEXT_ARCH = "clip-text"
CLIP_VISION_ARCH = "clip-vision"

class TensorNameMap:
    def __init__(self, n_text_blocks: int, n_vision_blocks: int, model_keys_path: Optional[Path] = None):
        self.mapping = {}

        # 文本模型的块级映射
        self.text_block_mappings = {
            gguf.constants.MODEL_TENSOR.ATTN_Q: ("text_model.encoder.layers.{bid}.self_attn.q_proj",),
            gguf.constants.MODEL_TENSOR.ATTN_K: ("text_model.encoder.layers.{bid}.self_attn.k_proj",),
            gguf.constants.MODEL_TENSOR.ATTN_V: ("text_model.encoder.layers.{bid}.self_attn.v_proj",),
            gguf.constants.MODEL_TENSOR.ATTN_OUT: ("text_model.encoder.layers.{bid}.self_attn.out_proj",),
            gguf.constants.MODEL_TENSOR.ATTN_NORM: ("text_model.encoder.layers.{bid}.layer_norm1",),
            gguf.constants.MODEL_TENSOR.ATTN_NORM_2: ("text_model.encoder.layers.{bid}.layer_norm2",),
            gguf.constants.MODEL_TENSOR.FFN_UP: ("text_model.encoder.layers.{bid}.mlp.fc1",),
            gguf.constants.MODEL_TENSOR.FFN_DOWN: ("text_model.encoder.layers.{bid}.mlp.fc2",),
        }

        # 视觉模型的块级映射
        self.vision_block_mappings = {
            gguf.constants.MODEL_TENSOR.ATTN_Q: ("vision_model.encoder.layers.{bid}.self_attn.q_proj",),
            gguf.constants.MODEL_TENSOR.ATTN_K: ("vision_model.encoder.layers.{bid}.self_attn.k_proj",),
            gguf.constants.MODEL_TENSOR.ATTN_V: ("vision_model.encoder.layers.{bid}.self_attn.v_proj",),
            gguf.constants.MODEL_TENSOR.ATTN_OUT: ("vision_model.encoder.layers.{bid}.self_attn.out_proj",),
            gguf.constants.MODEL_TENSOR.ATTN_NORM: ("vision_model.encoder.layers.{bid}.layer_norm1",),
            gguf.constants.MODEL_TENSOR.ATTN_NORM_2: ("vision_model.encoder.layers.{bid}.layer_norm2",),
            gguf.constants.MODEL_TENSOR.FFN_UP: ("vision_model.encoder.layers.{bid}.mlp.fc1",),
            gguf.constants.MODEL_TENSOR.FFN_DOWN: ("vision_model.encoder.layers.{bid}.mlp.fc2",),
        }

        # 添加文本模型块级映射
        for bid in range(n_text_blocks):
            for tensor_type, patterns in self.text_block_mappings.items():
                for pattern in patterns:
                    key = pattern.format(bid=bid)
                    self.mapping[key] = ("text", tensor_type)

        # 添加视觉模型块级映射
        for bid in range(n_vision_blocks):
            for tensor_type, patterns in self.vision_block_mappings.items():
                for pattern in patterns:
                    key = pattern.format(bid=bid)
                    self.mapping[key] = ("vision", tensor_type)

        # 添加文本模型全局映射
        self.text_global_mappings = {
            gguf.constants.MODEL_TENSOR.TOKEN_EMBD: ("text_model.embeddings.token_embedding",),
            gguf.constants.MODEL_TENSOR.POS_EMBD: ("text_model.embeddings.position_embedding",),
            gguf.constants.MODEL_TENSOR.OUTPUT_NORM: ("text_model.final_layer_norm",),
            gguf.constants.MODEL_TENSOR.T_PROJECTION: ("text_projection",),
        }

        # 添加视觉模型全局映射
        self.vision_global_mappings = {
            gguf.constants.MODEL_TENSOR.V_ENC_EMBD_PATCH: ("vision_model.embeddings.patch_embedding",),
            gguf.constants.MODEL_TENSOR.V_PROJECTION: ("visual_projection",),
            gguf.constants.MODEL_TENSOR.POS_EMBD: ("vision_model.embeddings.position_embedding",),
            gguf.constants.MODEL_TENSOR.OUTPUT_NORM: ("vision_model.post_layernorm",),
            gguf.constants.MODEL_TENSOR.INPUT_NORM: ("vision_model.pre_layrnorm",),
            gguf.constants.MODEL_TENSOR.CLS: ("vision_model.embeddings.class_embedding",),
        }

        for tensor_type, patterns in self.text_global_mappings.items():
            for pattern in patterns:
                self.mapping[pattern] = ("text", tensor_type)

        for tensor_type, patterns in self.vision_global_mappings.items():
            for pattern in patterns:
                self.mapping[pattern] = ("vision", tensor_type)

        # 如果提供了模型键映射文件，加载它以获取更准确的映射
        self.model_keys = None
        if model_keys_path is not None and model_keys_path.exists():
            with open(model_keys_path, "r") as f:
                self.model_keys = json.load(f)

    def get_mapping(self, key: str) -> Optional[Tuple[str, gguf.constants.MODEL_TENSOR]]:
        # 处理权重和偏置
        base_key = key.replace(".weight", "").replace(".bias", "")

        # 首先检查是否在直接映射中
        if base_key in self.mapping:
            return self.mapping[base_key]

        # 如果有模型键映射，尝试使用它
        if self.model_keys is not None:
            # 检查是否是模板键的实例
            for template_key, instances in self.model_keys.items():
                if key in instances:
                    # 从模板中提取bid
                    if "{bid}" in template_key:
                        # 找到层号
                        parts = key.split(".")
                        if "layers" in parts:
                            layer_idx = parts.index("layers")
                            if layer_idx + 1 < len(parts):
                                bid = int(parts[layer_idx + 1])
                                # 构造模板键并查找映射
                                base_template = template_key.replace(".weight", "").replace(".bias", "")
                                if base_template in self.mapping:
                                    return self.mapping[base_template]

        return None


class BGEClipConverter:
    def __init__(
        self,
        model_dir: Path,
        output_text_path: Path,
        output_vision_path: Path,
        ftype: int,
        model_keys_path: Optional[Path] = None,
    ):
        self.model_dir = model_dir
        self.output_text_path = output_text_path
        self.output_vision_path = output_vision_path
        self.ftype = ftype
        self.model_keys_path = model_keys_path
        self.config = self.load_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # 文本模型参数
        self.text_n_blocks = self.config["text_config"]["num_hidden_layers"]
        self.text_n_heads = self.config["text_config"]["num_attention_heads"]
        self.text_hidden_size = self.config["text_config"]["hidden_size"]
        self.text_intermediate_size = self.config["text_config"]["intermediate_size"]
        self.text_vocab_size = self.config["text_config"]["vocab_size"]
        self.text_max_position_embeddings = self.config["text_config"]["max_position_embeddings"]
        self.text_layer_norm_eps = self.config["text_config"]["layer_norm_eps"]

        # 视觉模型参数
        self.vision_n_blocks = self.config["vision_config"]["num_hidden_layers"]
        self.vision_n_heads = self.config["vision_config"]["num_attention_heads"]
        self.vision_hidden_size = self.config["vision_config"]["hidden_size"]
        self.vision_intermediate_size = self.config["vision_config"]["intermediate_size"]
        self.vision_image_size = self.config["vision_config"]["image_size"]
        self.vision_patch_size = self.config["vision_config"]["patch_size"]
        self.vision_layer_norm_eps = self.config["vision_config"]["layer_norm_eps"]

        # 投影维度
        self.projection_dim = self.config["projection_dim"]

        self.tensor_map = TensorNameMap(self.text_n_blocks, self.vision_n_blocks, model_keys_path)

        # 创建两个GGUF写入器，分别用于文本和视觉模型
        self.text_gguf_writer = gguf.GGUFWriter(output_text_path, CLIP_TEXT_ARCH)
        self.vision_gguf_writer = gguf.GGUFWriter(output_vision_path, CLIP_VISION_ARCH)

        # 用于存储已处理的张量
        self.text_tensors = {}
        self.vision_tensors = {}

    def load_config(self):
        config_path = self.model_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set_gguf_parameters(self):
        # 设置文本模型参数
        # self.text_gguf_writer.add_architecture()
        self.text_gguf_writer.add_context_length(self.text_max_position_embeddings)
        self.text_gguf_writer.add_embedding_length(self.text_hidden_size)
        self.text_gguf_writer.add_block_count(self.text_n_blocks)
        self.text_gguf_writer.add_layer_norm_eps(self.text_layer_norm_eps)
        self.text_gguf_writer.add_causal_attention(False)
        self.text_gguf_writer.add_tokenizer_model("bert")
        tokens = [0 for i in range(self.tokenizer.vocab_size)]
        for key, val in self.tokenizer.get_vocab().items():
            tokens[val] = key
        self.text_gguf_writer.add_token_list(tokens)
        self.text_gguf_writer.add_feed_forward_length(self.text_intermediate_size)
        self.text_gguf_writer.add_head_count(self.text_n_heads)
        self.text_gguf_writer.add_vocab_size(self.text_vocab_size)
        self.text_gguf_writer.add_file_type(self.ftype)
        self.text_gguf_writer.add_uint32("projection_dim", self.projection_dim)
        self.text_gguf_writer.add_eos_token_id(self.config["text_config"]["eos_token_id"])
        self.text_gguf_writer.add_bos_token_id(self.config["text_config"]["bos_token_id"])
        self.text_gguf_writer.add_logit_scale(self.config["logit_scale_init_value"])
        # self.text_gguf_writer.add_key_value(f"{CLIP_TEXT_ARCH}.hidden_size", self.config["text_config"]["hidden_size"], gguf.GGUFValueType.UINT32)

        # 设置视觉模型参数
        # self.vision_gguf_writer.add_architecture()
        self.vision_gguf_writer.add_feed_forward_length(self.vision_intermediate_size)
        self.vision_gguf_writer.add_head_count(self.vision_n_heads)
        self.vision_gguf_writer.add_block_count(self.vision_n_blocks)
        self.vision_gguf_writer.add_embedding_length(self.vision_hidden_size)
        self.vision_gguf_writer.add_layer_norm_eps(self.vision_layer_norm_eps)
        self.vision_gguf_writer.add_file_type(self.ftype)
        self.vision_gguf_writer.add_logit_scale(self.config["logit_scale_init_value"])
        self.vision_gguf_writer.add_context_length(int(self.vision_image_size / self.vision_patch_size) + 1)
        self.vision_gguf_writer.add_causal_attention(False)
        self.vision_gguf_writer.add_tokenizer_model("no_vocab")
        self.vision_gguf_writer.add_uint32("projection_dim", self.projection_dim)

        self.vision_gguf_writer.add_uint32("image_size", self.vision_image_size)
        self.vision_gguf_writer.add_uint32("patch_size", self.vision_patch_size)
        # self.vision_gguf_writer.add_key_value(f"{CLIP_VISION_ARCH}.hidden_size", self.config["vision_config"]["hidden_size"], gguf.GGUFValueType.UINT32)

    def convert_tensor_name(self, name: str) -> Tuple[Optional[str], str]:
        mapping = self.tensor_map.get_mapping(name)
        if mapping is None:
            return None, None

        model_type, tensor_type = mapping
        bid = None
        parts = name.split(".")
        if "layers" in parts:
            layer_idx = parts.index("layers")
            if layer_idx + 1 < len(parts):
                bid = int(parts[layer_idx + 1])

        # 处理特殊后缀
        suffix = ""
        if name.endswith(".weight"):
            suffix = ".weight"
        elif name.endswith(".bias"):
            suffix = ".bias"

        # 构造GGUF tensor名称
        if isinstance(tensor_type, gguf.constants.MODEL_TENSOR):
            if bid is not None:
                return model_type, f"{gguf.constants.TENSOR_NAMES[tensor_type].format(bid=bid)}{suffix}"
            return model_type, f"{gguf.constants.TENSOR_NAMES[tensor_type]}{suffix}"
        else:
            # 处理特殊情况，如vision_model.pre_layrnorm
            return model_type, f"{tensor_type.split('.')[-1].lower()}{suffix}"

    def convert(self):
        self.set_gguf_parameters()

        # 加载模型权重
        model_files = list(self.model_dir.glob("*.bin")) + list(self.model_dir.glob("*.safetensors"))

        for file in model_files:
            if file.suffix == ".safetensors":
                from safetensors import safe_open

                with safe_open(file, framework="pt") as f:
                    tensors = f.keys()
                    for key in tensors:
                        tensor = f.get_tensor(key)
                        self.process_tensor(key, tensor)
            else:
                state_dict = torch.load(file, map_location="cpu")
                for key, tensor in state_dict.items():
                    self.process_tensor(key, tensor)

        # 写入文本模型
        self.text_gguf_writer.write_header_to_file(path=self.output_text_path)
        self.text_gguf_writer.write_kv_data_to_file()
        self.text_gguf_writer.write_tensors_to_file(progress=True)
        self.text_gguf_writer.close()

        # 写入视觉模型
        self.vision_gguf_writer.write_header_to_file(path=self.output_vision_path)
        self.vision_gguf_writer.write_kv_data_to_file()
        self.vision_gguf_writer.write_tensors_to_file(progress=True)
        self.vision_gguf_writer.close()

    def process_tensor(self, name: str, tensor: torch.Tensor):
        model_type, gguf_name = self.convert_tensor_name(name)

        if model_type is None or gguf_name is None:
            print(f"Skipping tensor: {name}, not mapped")
            return

        # 根据 ftype 参数转换为相应的数据类型
        ftype = gguf.GGMLQuantizationType.F32
        if self.ftype == gguf.GGMLQuantizationType.F16 and len(tensor.shape) >= 2:
            tensor = tensor.to(torch.float16)
            ftype = gguf.GGMLQuantizationType.F16
        elif self.ftype == gguf.GGMLQuantizationType.F32:
            tensor = tensor.to(torch.float32)
        else:
            tensor = tensor.to(torch.float32)
        print(f"Converting tensor: {name} => {model_type}: {gguf_name} | type: {tensor.dtype}")

        # 根据模型类型选择相应的GGUF写入器
        if model_type == "text":
            self.text_gguf_writer.add_tensor(gguf_name, tensor.numpy(), raw_shape=tensor.shape, raw_dtype=ftype)
            self.text_tensors[gguf_name] = tensor
        elif model_type == "vision":
            self.vision_gguf_writer.add_tensor(gguf_name, tensor.numpy(), raw_shape=tensor.shape, raw_dtype=ftype)
            self.vision_tensors[gguf_name] = tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BGE-VL model to GGUF format")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to BGE-VL model directory")
    parser.add_argument("--text-outfile", type=str, required=True, help="Output GGUF file path for text model")
    parser.add_argument("--vision-outfile", type=str, required=True, help="Output GGUF file path for vision model")
    parser.add_argument("--model-keys", type=str, help="Path to model keys JSON file")
    parser.add_argument("--ftype", type=int, default=1, choices=[0, 1], help="File type (0=f32, 1=f16)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_keys_path = Path(args.model_keys) if args.model_keys else None
    converter = BGEClipConverter(
        model_dir=Path(args.model_dir),
        output_text_path=Path(args.text_outfile),
        output_vision_path=Path(args.vision_outfile),
        ftype=args.ftype,
        model_keys_path=model_keys_path,
    )
    converter.convert()
    print(f"Successfully converted text model to {args.text_outfile}, vision model to {args.vision_outfile}")
