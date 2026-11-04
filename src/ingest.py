# -*- coding: utf-8 -*-
"""构建知识库：课程课件 PDF → 文本提取 → 切分 → 向量化 → 存入 Chroma。

用法：
    python src/ingest.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

COLLECTION_NAME = "nlp_course"


def extract_pdf_text(pdf_path):
    """提取单个 PDF 的全部页文本，返回 [(页码, 文本), ...]。"""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            pages.append((i + 1, text))
    return pages


def split_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """把长文本切成带重叠的块，尽量在句子边界断开。"""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    seps = ["。", "！", "？", "；", "! ", "? ", ". ", "; ", "\n", " "]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in seps:
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def main():
    print("加载 embedding 模型：%s" % config.EMBEDDING_MODEL)
    embedder = SentenceTransformer(
        config.EMBEDDING_MODEL,
        cache_folder=str(config.EMBEDDING_DIR),
    )

    client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))
    # 重建知识库：先删旧集合
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAME)

    pdf_files = sorted(config.DOCS_DIR.glob("*.pdf"))
    print("待处理文档 %d 个：" % len(pdf_files))

    total = 0
    for pdf in pdf_files:
        pages = extract_pdf_text(pdf)
        n_chunks = 0
        for page_no, text in pages:
            chunks = split_text(text)
            if not chunks:
                continue
            embs = embedder.encode(chunks, normalize_embeddings=True,
                                   show_progress_bar=False)
            collection.add(
                ids=["c%d" % (total + j) for j in range(len(chunks))],
                embeddings=[e.tolist() for e in embs],
                documents=chunks,
                metadatas=[{"source": pdf.name, "page": page_no} for _ in chunks],
            )
            total += len(chunks)
            n_chunks += len(chunks)
        print("  %s：%d 页 -> %d 块" % (pdf.name, len(pages), n_chunks))

    print("完成。向量库共 %d 个文本块，存于 %s" % (collection.count(),
                                                 config.VECTOR_STORE_DIR))


if __name__ == "__main__":
    main()
