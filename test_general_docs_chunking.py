"""
일반 문서(장-절-조 구조 없음)에 대한 청킹 및 페이지 검증
"""

import sys
sys.path.insert(0, '/home/coder/project/data_job/src/chunking')
from LegalDocumentProcessor import LegalParserConfig, DocumentTask, LegalDocumentProcessor


def run_general_doc_chunking(test_case: str = "headers_3per_page"):
    """
    일반 문서 청킹 실행

    Args:
        test_case: "headers_3per_page", "headers_5per_page", "lines_50per_page"
    """
    print(f"\n{'='*80}")
    print(f"일반 문서 청킹 실행: {test_case}")
    print(f"{'='*80}\n")

    # 설정
    config = LegalParserConfig(
        TOKENIZER_FILE_PATH="./tokenizer.json",
        LEAF_TOKEN_SIZE=512,
        LEAF_TOKEN_OVERLAP=50,
        PARENT_TARGET_TOKEN_SIZE=1500,
        PARENT_MAX_TOKEN_SIZE=2048,
        OUTPUT_DIR=f"/home/coder/project/data_job/output/general_docs_{test_case}"
    )

    tasks = [
        DocumentTask(
            source=f"/home/coder/project/data_job/data/test_general_docs/{test_case}",
            category="일반문서",
            collection_name="general_docs_test",
            file_type="문서",
            recursive=False
        ),
    ]

    # 청킹 실행
    processor = LegalDocumentProcessor(config=config, test_mode=True)
    processor.run(tasks)

    print(f"\n✅ 청킹 완료: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    # 모든 테스트 케이스 실행
    for test_case in ["headers_3per_page", "headers_5per_page", "lines_50per_page"]:
        run_general_doc_chunking(test_case)
