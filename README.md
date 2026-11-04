# 课程设计 实验三：检索增强生成（RAG）系统

本项目完成《自然语言处理课程设计》实验三，构建一个面向本专业课程资料的**检索增强生成（RAG）知识问答系统**。

- 任务：围绕《自然语言处理》课程课件（大语言模型预训练与适配），实现"检索 + 生成"的问答系统
- 核心：本地向量检索（中文 embedding + Chroma）+ 本地大模型生成（Qwen2.5-3B-Instruct），全程不依赖外部 API
- 所有模型、缓存、向量库均位于本目录（E 盘），不写入 C 盘

---

## 1. 目录结构

```
课程设计_实验三_RAG/
├─ .venv/                     # Python 3.8 虚拟环境
├─ _cache/                    # pip / HF / modelscope 缓存（E 盘）
├─ config.py                  # 路径、环境变量、模型名、检索/生成参数
├─ requirements.txt
├─ src/
│  ├─ download_models.py      # 下载生成模型 Qwen2.5-3B-Instruct 到本地
│  ├─ ingest.py               # 建库：PDF → 提取 → 切分 → 向量化 → 存入 Chroma
│  ├─ rag.py                  # RAG 核心：embedding 检索 + 大模型生成
│  └─ ask.py                  # 命令行交互式问答入口
├─ data/
│  ├─ docs/                   # 知识库源文档（6 个课程课件 PDF）
│  └─ vector_store/           # Chroma 向量库
├─ models/
│  ├─ embedding/              # bge-small-zh-v1.5 中文向量模型
│  └─ llm/                    # Qwen2.5-3B-Instruct 生成模型
└─ output/                    # 运行日志 / 图表
```

## 2. 环境

- Python 3.8（`C:\Users\86152\AppData\Local\Programs\Python\Python38`，已建 `.venv`）
- RTX 4060 Laptop 8GB（CPU 也可运行，只是更慢）
- 依赖：torch 2.4.1+cu121、transformers 4.46、sentence-transformers 3.2、chromadb 0.5、pypdf

> 环境注意：本机系统默认 Python（Anaconda）SSL 损坏，需用上述 Python38 建环境；
> chromadb 依赖的 posthog 需降级到 `posthog<3.0` 才能在 Python 3.8 下导入
> （`pip install "posthog<3.0"`）。

## 3. 运行

```bash
cd /d E:\自然语言处理\课程设计_实验三_RAG

# 1) 下载生成模型（约 6.2 GB，走 hf-mirror，只下一次）
.venv\Scripts\python.exe src\download_models.py

# 2) 建向量库（首次会自动下载 embedding 模型，约 95 MB）
.venv\Scripts\python.exe src\ingest.py

# 3) 交互式问答
.venv\Scripts\python.exe src\ask.py
```

`config.py` 会在导入第三方库之前设置 `HF_HOME`、`HF_ENDPOINT`、`TORCH_HOME`、
`TMP` 等，保证运行时也不写 C 盘。

## 4. 技术方案

| 环节 | 实现 |
|---|---|
| 文档加载 | pypdf 提取课件 PDF 文本 |
| 文本切分 | 按字符切分（500 字 / 50 字重叠），优先在句子边界断开 |
| 向量化 | `BAAI/bge-small-zh-v1.5`（中文，512 维） |
| 向量库 | Chroma 本地持久化 |
| 检索 | 余弦相似度 Top-K（默认 4） |
| 生成 | `Qwen/Qwen2.5-3B-Instruct`（float16，8GB 显存直跑） |

问答流程：用户问题 → embedding 向量化 → 在向量库检索 Top-K 相关片段 → 拼接
系统提示 + 检索片段 + 问题 → 大模型生成回答（严格依据检索到的资料，不编造）。

## 5. 知识库

6 个《自然语言处理》课程课件（共 265 个文本块）：

- 第8章 - 大语言模型的预训练.pdf
- 07-预训练语言模型进阶-part2.pdf
- 09-大语言模型的适配 - RLHF.pdf
- 09-大语言模型的适配-part3.pdf
- 09-大语言模型的适配-多任务指令微调.pdf
- 09-大语言模型的适配-模型蒸馏与裁剪.pdf

## 6. 答辩建议

- 展示 `src/ingest.py` 建库日志（6 文档 → 265 块）
- 展示 `src/ask.py` 问答效果：可提问"什么是 RLHF？""什么是 LoRA？""知识蒸馏是什么？"
  等，系统会给出依据资料的回答并列出检索到的片段来源（文件名 + 页码）
- 讲解 RAG 的"检索 + 生成"两阶段流程，以及为什么要检索增强（避免大模型幻觉）
