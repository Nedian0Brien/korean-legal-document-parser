#!/usr/bin/env python3
"""
마크다운 헤더 구분자 추가 후 일반 문서 청킹 테스트
"""
import sys
sys.path.insert(0, '/home/coder/project/data_job/src/chunking')

from LegalDocumentProcessor import (
    LegalDocumentProcessor,
    LegalParserConfig,
    DocumentTask
)

if __name__ == "__main__":
    print("=" * 80)
    print("마크다운 헤더 구분자 추가 후 일반 문서 청킹 테스트")
    print("=" * 80)

    # 설정 초기화
    config = LegalParserConfig(
        TOKENIZER_FILE_PATH="./tokenizer.json",
        LEAF_TOKEN_SIZE=512,
        LEAF_TOKEN_OVERLAP=50,
        PARENT_TARGET_TOKEN_SIZE=1500,
        PARENT_MAX_TOKEN_SIZE=2048,
        OUTPUT_DIR="/home/coder/project/data_job/output_markdown_test"
    )

    # 테스트할 일반 문서 작업 정의
    tasks = [
        DocumentTask(
            source="/home/coder/project/data_job/data/md",
            category="일반문서",
            collection_name="test_markdown_docs",
            file_type="마크다운문서",
            recursive=False
        ),
    ]

    # 프로세서 실행 (테스트 모드)
    processor = LegalDocumentProcessor(config=config, test_mode=True)

    print("\n📝 테스트 대상:")
    print(f"  - 디렉토리: {tasks[0].source}")
    print(f"  - 카테고리: {tasks[0].category}")
    print(f"  - 출력 경로: {config.OUTPUT_DIR}")
    print(f"\n⚙️ 청킹 설정:")
    print(f"  - LEAF: {config.LEAF_TOKEN_SIZE} tokens (overlap: {config.LEAF_TOKEN_OVERLAP})")
    print(f"  - PARENT: {config.PARENT_TARGET_TOKEN_SIZE} ~ {config.PARENT_MAX_TOKEN_SIZE} tokens")
    print(f"  - 구분자: Markdown 헤더 (\\n# ~ \\n######) + 문단 (\\n\\n) + 기타\n")

    processor.run(tasks)

    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)
    print(f"\n📂 결과 확인:")
    print(f"  {config.OUTPUT_DIR}/test_markdown_docs/일반문서/")
    print(f"    - *_structured.md : 트리 구조 및 Parent-Leaf 관계 시각화")
    print(f"    - *_chunks.json : 청크 메타데이터 및 relationships\n")
