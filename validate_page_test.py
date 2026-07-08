"""
페이지 마커가 있는 테스트 데이터의 청킹 결과 검증
"""

import sys
sys.path.insert(0, '/home/coder/project/data_job')

from test_page_validation import PageValidator


def validate_test_case(test_case: str = "3articles_per_page"):
    """테스트 케이스 검증"""
    print(f"\n{'#'*80}")
    print(f"# 검증: {test_case}")
    print(f"{'#'*80}\n")

    validator = PageValidator(
        data_dir=f"/home/coder/project/data_job/data/test_with_page_markers/{test_case}",
        output_dir=f"/home/coder/project/data_job/output/page_test_{test_case}/page_validation_test"
    )

    results = validator.run_validation()
    return results


if __name__ == "__main__":
    # 테스트 케이스들 검증
    for test_case in ["2articles_per_page", "3articles_per_page", "5articles_per_page"]:
        validate_test_case(test_case)
