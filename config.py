# -*- coding: utf-8 -*-
"""实验三：检索增强生成（RAG）系统 —— 全局配置。

所有模型、缓存、向量库都指向项目目录（E 盘），不写 C 盘。
本文件必须在 import torch / transformers / sentence_transformers 之前导入。
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "_cache"
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"               # 知识库源文档（课程课件 PDF）
VECTOR_STORE_DIR = DATA_DIR / "vector_store"  # Chroma 向量库
MODEL_DIR = BASE_DIR / "models"            # 本地模型缓存
EMBEDDING_DIR = MODEL_DIR / "embedding"    # 向量化模型
LLM_DIR = MODEL_DIR / "llm"                # 生成模型
OUTPUT_DIR = BASE_DIR / "output"

# ---------------------------------------------------------------------------
# 必须在导入第三方库之前设置的环境变量（保证模型/缓存不写 C 盘）
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = str(CACHE_DIR / "hf")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
os.environ["TMP"] = str(CACHE_DIR / "tmp")
os.environ["TEMP"] = str(CACHE_DIR / "tmp")
os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUTF8", "1")
# 绕过失效的系统代理（Windows 注册表中的代理关闭后仍会被 requests 读取）
for _key in ("no_proxy", "NO_PROXY"):
    os.environ.setdefault(_key, "*")

for _p in (CACHE_DIR, DATA_DIR, DOCS_DIR, VECTOR_STORE_DIR, MODEL_DIR,
           EMBEDDING_DIR, LLM_DIR, OUTPUT_DIR,
           CACHE_DIR / "tmp", CACHE_DIR / "matplotlib", OUTPUT_DIR / "figures"):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
# 中文向量化模型（把文档和问题映射到同一向量空间）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# 生成模型（根据检索到的上下文回答中文问题）
# 说明：8GB 显存本地部署，采用 3B 版 float16 直跑（约 6GB 显存、无需额外量化库）。
# 若想换更强的 7B 版，改成 "Qwen/Qwen2.5-7B-Instruct" 并配合 4-bit 量化（需 bitsandbytes）。
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ---------------------------------------------------------------------------
# 文档切分与检索参数
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500       # 每个文本块的最大字符数
CHUNK_OVERLAP = 50     # 相邻块的重叠字符数
TOP_K = 4              # 每次问答检索的文档块数量

# ---------------------------------------------------------------------------
# 生成参数
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
