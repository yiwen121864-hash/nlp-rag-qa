# -*- coding: utf-8 -*-
"""RAG 问答核心：向量检索 + 大模型生成。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb

COLLECTION_NAME = "nlp_course"

SYSTEM_PROMPT = (
    "你是一名《自然语言处理》课程助教。请严格依据下面提供的【参考资料】回答用户问题。\n"
    "要求：\n"
    "1. 只使用资料中明确出现的内容，不要编造；\n"
    "2. 如果资料中没有答案，直接回答“资料中没有相关内容”，不要猜测；\n"
    "3. 回答使用中文，条理清晰，必要时分点。"
)


class RAGSystem:
    def __init__(self):
        print("  - 加载 embedding 模型 ...")
        self.embedder = SentenceTransformer(
            config.EMBEDDING_MODEL,
            cache_folder=str(config.EMBEDDING_DIR),
        )

        print("  - 打开向量库 ...")
        client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)

        print("  - 加载生成模型 %s ..." % config.LLM_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(str(config.LLM_DIR))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(config.LLM_DIR),
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("  - 就绪")

    def retrieve(self, question, top_k=config.TOP_K):
        """检索与问题最相关的文档块。"""
        q_emb = self.embedder.encode([question], normalize_embeddings=True)[0].tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [(d, m) for d, m in zip(docs, metas)]

    def answer(self, question):
        """检索 + 生成，返回 (回答文本, 检索片段列表)。"""
        hits = self.retrieve(question)
        context = "\n\n".join("[%d] %s" % (i + 1, d) for i, (d, _) in enumerate(hits))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "【参考资料】\n%s\n\n【问题】\n%s" % (context, question)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        return answer, hits
