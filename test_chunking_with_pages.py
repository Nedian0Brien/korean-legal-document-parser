"""
페이지 마커가 있는 문서로 청킹을 실행하고 검증하는 통합 스크립트
"""

import sys
import os

# LegalDocumentProcessor 임포트
sys.path.insert(0, '/home/coder/project/data_job/src/chunking')
from LegalDocumentProcessor import LegalParserConfig, DocumentTask, LegalDocumentProcessor


def run_chunking_test(test_case: str = "3articles_per_page"):
    """
    페이지 마커가 있는 테스트 데이터로 청킹 실행

    Args:
        test_case: "2articles_per_page", "3articles_per_page", "5articles_per_page"
    """
    print(f"\n{'='*80}")
    print(f"청킹 실행: {test_case}")
    print(f"{'='*80}\n")

    # 설정
    config = LegalParserConfig(
        TOKENIZER_FILE_PATH="./tokenizer.json",
        LEAF_TOKEN_SIZE=512,
        LEAF_TOKEN_OVERLAP=50,
        PARENT_TARGET_TOKEN_SIZE=1500,
        PARENT_MAX_TOKEN_SIZE=2048,
        OUTPUT_DIR=f"/home/coder/project/data_job/output/page_test_{test_case}"
    )

    tasks = [
        DocumentTask(
            source=f"/home/coder/project/data_job/data/test_with_page_markers/{test_case}",
            category="법령",
            collection_name="page_validation_test",
            file_type="법령",
            recursive=True
        ),
    ]

    # 청킹 실행
    processor = LegalDocumentProcessor(config=config, test_mode=True)
    processor.run(tasks)

    print(f"\n✅ 청킹 완료: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    # 모든 테스트 케이스 실행
    for test_case in ["2articles_per_page", "3articles_per_page", "5articles_per_page"]:
        run_chunking_test(test_case)
