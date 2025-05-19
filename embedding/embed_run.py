"""
전체 임베딩 파이프라인을 한 번에 실행합니다.

순서
1. fullgraph/build_graph.py  →  전체 그래프 + Chroma 임베딩
2. subgraph/build_subgraph.py →  서브그래프 + Chroma 임베딩

사용법:
    python -m embedding.embed_run
또는
    python embedding/embed_run.py
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FULL_SCRIPT = BASE_DIR / "fullgraph" / "build_fullgraph.py"
SUB_SCRIPT = BASE_DIR / "subgraph" / "build_subgraph.py"


def run(step: str, script: Path) -> None:
    """단일 스텝 실행 + 경과시간 출력"""
    start = time.time()
    print(f"\n [{step}] start → {module_path}")
    subprocess.run([sys.executable, "-m", module_path], check=True)
    elapsed = time.time() - start
    print(f" [{step}] done  ({elapsed:,.1f}s)")


def main() -> None:
    """파이프라인 엔트리포인트"""
    run("FULLGRAPH", "embedding.fullgraph.build_fullgraph")
    print("\n 전체 그래프·벡터 임베딩 완료되었습니다!")
    print("\n 서브 그래프·벡터 임베딩 작업을 시작하겠습니다!")
    run("SUBGRAPH", "embedding.subgraph.build_subgraph")
    print("\n모든 작업을 완료되었습니다!")


if __name__ == "__main__":
    main()
