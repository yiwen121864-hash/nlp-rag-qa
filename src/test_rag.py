# -*- coding: utf-8 -*-
"""端到端测试：加载 RAG 系统，回答几个预设问题，验证检索 + 生成全流程。

用法：
    python src/test_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根（config）
sys.path.insert(0, str(Path(__file__).resolve().parent))          # src（rag）

import config
from rag import RAGSystem

QUESTIONS = [
    "什么是RLHF？",
    "什么是LoRA？",
    "知识蒸馏是什么？",
]


def main():
    print("正在加载 RAG 系统（embedding + 向量库 + 生成模型）...")
    rag = RAGSystem()
    print("加载完成。\n")

    for q in QUESTIONS:
        print("=" * 64)
        print("问题：%s" % q)
        answer, hits = rag.answer(q)
        print("\n回答：\n%s" % answer)
        print("\n依据的检索片段：")
        for i, (doc, meta) in enumerate(hits):
            print("  [%d] %s 第 %s 页" % (i + 1, meta.get("source"), meta.get("page")))
        print()


if __name__ == "__main__":
    main()
