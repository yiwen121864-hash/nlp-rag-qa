# -*- coding: utf-8 -*-
"""命令行交互式问答入口。

用法：
    python src/ask.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from rag import RAGSystem


def main():
    rag = RAGSystem()
    print("\n=== 检索增强生成（RAG）问答系统 ===")
    print("知识库：课程课件（大语言模型预训练与适配）")
    print("输入问题开始，输入 exit / 退出 结束。\n")

    while True:
        try:
            q = input("问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出", "q"):
            break

        print("检索中 ...")
        answer, hits = rag.answer(q)
        print("\n【回答】\n%s" % answer)
        print("\n【依据的检索片段】")
        for i, (doc, meta) in enumerate(hits):
            print("  %d. 来源 %s（第 %s 页）" % (i + 1, meta.get("source"), meta.get("page")))
            print("     %s ..." % doc[:80])
        print()


if __name__ == "__main__":
    main()
