# -*- coding: utf-8 -*-
"""下载生成模型到本地（E 盘项目目录），供 rag.py 加载。

用法：
    python src/download_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from huggingface_hub import snapshot_download


def main():
    print("下载生成模型 %s" % config.LLM_MODEL)
    print("目标目录：%s" % config.LLM_DIR)
    snapshot_download(
        repo_id=config.LLM_MODEL,
        local_dir=str(config.LLM_DIR),
        local_dir_use_symlinks=False,
    )
    print("完成：%s" % config.LLM_DIR)


if __name__ == "__main__":
    main()
