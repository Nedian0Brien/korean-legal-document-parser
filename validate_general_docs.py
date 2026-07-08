"""
일반 문서의 페이지 메타데이터 검증
"""

import sys
sys.path.insert(0, '/home/coder/project/data_job')

from test_page_validation import PageValidator


def validate_general_docs(test_case: str):
    """일반 문서 테스트 케이스 검증"""
    print(f"\n{'#'*80}")
    print(f"# 검증: {test_case}")
    print(f"{'#'*80}\n")

    validator = PageValidator(
        data_dir=f"/home/coder/project/data_job/data/test_general_docs/{test_case}",
        output_dir=f"/home/coder/project/data_job/output/general_docs_{test_case}/general_docs_test"
    )

    results = validator.run_validation()
    return results


if __name__ == "__main__":
    # 테스트 케이스들 검증
    all_results = {}
    for test_case in ["headers_3per_page", "headers_5per_page", "lines_50per_page"]:
        results = validate_general_docs(test_case)
        all_results[test_case] = results

    # 전체 요약
    print(f"\n\n{'='*80}")
    print("전체 테스트 케이스 비교")
    print(f"{'='*80}\n")

    print(f"{'테스트 케이스':<30} {'총 청크':>10} {'일치율':>10} {'불일치':>10}")
    print(f"{'-'*80}")

    for test_case, results in all_results.items():
        if results:
            total = sum(r['total_chunks'] for r in results)
            valid = sum(r['valid'] for r in results)
            invalid = sum(r['invalid'] for r in results)
            accuracy = (valid / total * 100) if total > 0 else 0

            print(f"{test_case:<30} {total:>10} {accuracy:>9.1f}% {invalid:>10}")
