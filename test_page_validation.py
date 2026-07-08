"""
법령 청킹 결과의 페이지 메타데이터 검증 스크립트
- 청크의 실제 내용 위치와 메타데이터의 페이지가 일치하는지 확인
- {n}--------- 형식의 페이지 마커를 기준으로 검증
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class PageValidator:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.page_marker_pattern = re.compile(r'\{(\d+)\}[-]+')

    def find_md_file(self, json_filename: str) -> Optional[Path]:
        """JSON 파일명에서 원본 MD 파일 찾기"""
        # JSON 파일명에서 _chunks.json 제거하고 .md 추가
        base_name = json_filename.replace('_chunks.json', '.md')

        # data_dir에서 재귀적으로 검색
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file == base_name:
                    return Path(root) / file
        return None

    def parse_md_pages(self, md_path: Path) -> List[Tuple[int, int, str]]:
        """
        MD 파일을 파싱하여 페이지별 텍스트 범위 추출

        Returns:
            List of (page_num, start_char, end_char, content)
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 페이지 마커 찾기
        pages = []
        current_page = 1  # 첫 페이지는 1 (마커 전까지)
        current_start = 0

        for match in self.page_marker_pattern.finditer(content):
            # 현재 페이지의 끝 = 마커 시작 위치
            page_end = match.start()
            page_content = content[current_start:page_end]

            pages.append({
                'page': current_page,
                'start': current_start,
                'end': page_end,
                'content': page_content.strip()
            })

            # 다음 페이지 준비 (0-based 마커를 1-based로 변환)
            current_page = int(match.group(1)) + 1
            current_start = match.end()

        # 마지막 페이지 추가
        if current_start < len(content):
            pages.append({
                'page': current_page,
                'start': current_start,
                'end': len(content),
                'content': content[current_start:].strip()
            })

        return pages, content

    def extract_pure_text(self, text: str) -> str:
        """
        청크 텍스트에서 순수 본문만 추출
        - [계층 헤더] 제거
        - 공백/줄바꿈 정규화
        """
        # Contextual header 제거 ([...]\n)
        if text.strip().startswith('['):
            bracket_end = text.find(']\n')
            if bracket_end > 0:
                text = text[bracket_end + 2:]

        # 마크다운 헤더 제거
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        # 연속된 공백/줄바꿈 정규화
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def find_text_in_pages(self, chunk_text: str, pages: List[Dict], original_content: str) -> List[int]:
        """
        청크 텍스트가 어떤 페이지들에 걸쳐있는지 찾기

        Returns:
            List of page numbers where the chunk appears
        """
        # 순수 본문만 추출
        pure_chunk = self.extract_pure_text(chunk_text)

        # 너무 짧은 텍스트는 검증 불가
        if len(pure_chunk) < 20:
            return []

        # 청크의 시작/끝 스니펫 추출 (앞뒤 50자)
        start_snippet = pure_chunk[:50].strip()
        end_snippet = pure_chunk[-50:].strip()

        # 원본 콘텐츠에서 페이지 마커 제거
        clean_content = self.page_marker_pattern.sub('', original_content)
        clean_content = re.sub(r'\s+', ' ', clean_content)

        # 원본에서 청크 위치 찾기
        start_pos = clean_content.find(start_snippet)

        if start_pos == -1:
            # 정확한 매칭 실패 시, 더 짧은 스니펫으로 재시도
            start_snippet = pure_chunk[:30].strip()
            start_pos = clean_content.find(start_snippet)

        if start_pos == -1:
            return []  # 찾을 수 없음

        # 청크가 걸쳐있는 페이지 찾기
        chunk_len = len(pure_chunk)
        end_pos = start_pos + chunk_len

        found_pages = []
        for page_info in pages:
            # 페이지 마커 제거한 페이지 콘텐츠
            page_content_clean = self.page_marker_pattern.sub('', page_info['content'])
            page_content_clean = re.sub(r'\s+', ' ', page_content_clean)

            # 청크의 일부가 이 페이지에 있는지 확인
            if start_snippet in page_content_clean or end_snippet in page_content_clean:
                found_pages.append(page_info['page'])

        return found_pages

    def validate_chunk(self, chunk: Dict, pages: List[Dict], original_content: str) -> Dict:
        """단일 청크 검증"""
        chunk_id = chunk.get('id', 'unknown')
        chunk_text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        chunk_type = metadata.get('chunk_type', 'unknown')
        meta_page = metadata.get('page', '0')
        hierarchy = metadata.get('hierarchy', 'N/A')

        # 메타데이터 페이지를 정수로 변환
        try:
            meta_page_num = int(meta_page)
        except:
            meta_page_num = 0

        # 실제 페이지 찾기
        actual_pages = self.find_text_in_pages(chunk_text, pages, original_content)

        # 검증 결과
        is_valid = meta_page_num in actual_pages if actual_pages else None

        return {
            'chunk_id': chunk_id,
            'chunk_type': chunk_type,
            'hierarchy': hierarchy,
            'metadata_page': meta_page_num,
            'actual_pages': actual_pages,
            'is_valid': is_valid,
            'text_preview': self.extract_pure_text(chunk_text)[:100]
        }

    def validate_file(self, json_path: Path) -> Dict:
        """JSON 파일의 모든 청크 검증"""
        print(f"\n{'='*80}")
        print(f"검증 중: {json_path.name}")
        print(f"{'='*80}")

        # 원본 MD 파일 찾기
        md_path = self.find_md_file(json_path.name)
        if not md_path:
            print(f"⚠️  원본 MD 파일을 찾을 수 없습니다: {json_path.name}")
            return None

        print(f"✓ 원본 파일: {md_path}")

        # MD 파일 파싱
        pages, original_content = self.parse_md_pages(md_path)
        print(f"✓ 총 페이지 수: {len(pages)}")

        # JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"✓ 총 청크 수: {len(chunks)}")

        # 각 청크 검증
        results = []
        valid_count = 0
        invalid_count = 0
        unknown_count = 0

        for chunk in chunks:
            result = self.validate_chunk(chunk, pages, original_content)
            results.append(result)

            if result['is_valid'] is True:
                valid_count += 1
            elif result['is_valid'] is False:
                invalid_count += 1
            else:
                unknown_count += 1

        # 통계 출력
        print(f"\n📊 검증 결과:")
        print(f"  ✅ 일치: {valid_count}개 ({valid_count/len(chunks)*100:.1f}%)")
        print(f"  ❌ 불일치: {invalid_count}개 ({invalid_count/len(chunks)*100:.1f}%)")
        print(f"  ⚠️  검증 불가: {unknown_count}개 ({unknown_count/len(chunks)*100:.1f}%)")

        # 불일치 사례 출력
        if invalid_count > 0:
            print(f"\n❌ 불일치 사례 (상위 5개):")
            invalid_cases = [r for r in results if r['is_valid'] is False][:5]
            for i, case in enumerate(invalid_cases, 1):
                print(f"\n  [{i}] Chunk ID: {case['chunk_id'][:16]}...")
                print(f"      타입: {case['chunk_type']}")
                print(f"      계층: {case['hierarchy'][:80]}...")
                print(f"      메타데이터 페이지: {case['metadata_page']}")
                print(f"      실제 페이지: {case['actual_pages']}")
                print(f"      텍스트 미리보기: {case['text_preview']}...")

        return {
            'file': json_path.name,
            'md_file': md_path.name,
            'total_pages': len(pages),
            'total_chunks': len(chunks),
            'valid': valid_count,
            'invalid': invalid_count,
            'unknown': unknown_count,
            'results': results
        }

    def run_validation(self):
        """모든 법령 JSON 파일 검증"""
        # 법령 관련 JSON 파일 찾기
        json_files = []

        # 여러 가능한 경로 시도
        possible_paths = [
            Path(self.output_dir) / "rag_legal_docs_v1" / "법령",
            Path(self.output_dir) / "법령",
            Path(self.output_dir)
        ]

        law_output_dir = None
        for path in possible_paths:
            if path.exists():
                law_output_dir = path
                break

        if not law_output_dir:
            print(f"❌ 출력 디렉토리를 찾을 수 없습니다. 시도한 경로:")
            for path in possible_paths:
                print(f"  - {path}")
            return

        for json_file in law_output_dir.rglob("*_chunks.json"):
            json_files.append(json_file)

        print(f"🔍 검증할 파일 수: {len(json_files)}")

        # 각 파일 검증
        all_results = []
        for json_path in json_files:
            result = self.validate_file(json_path)
            if result:
                all_results.append(result)

        # 전체 통계
        print(f"\n\n{'='*80}")
        print(f"전체 검증 결과 요약")
        print(f"{'='*80}")

        total_chunks = sum(r['total_chunks'] for r in all_results)
        total_valid = sum(r['valid'] for r in all_results)
        total_invalid = sum(r['invalid'] for r in all_results)
        total_unknown = sum(r['unknown'] for r in all_results)

        print(f"\n📁 검증한 파일: {len(all_results)}개")
        print(f"📄 총 청크 수: {total_chunks}개")
        print(f"\n정확도:")
        print(f"  ✅ 일치: {total_valid}개 ({total_valid/total_chunks*100:.1f}%)")
        print(f"  ❌ 불일치: {total_invalid}개 ({total_invalid/total_chunks*100:.1f}%)")
        print(f"  ⚠️  검증 불가: {total_unknown}개 ({total_unknown/total_chunks*100:.1f}%)")

        # 파일별 상세 결과
        print(f"\n\n파일별 상세 결과:")
        print(f"{'파일명':<60} {'청크수':>8} {'일치율':>8}")
        print(f"{'-'*80}")
        for r in all_results:
            accuracy = r['valid'] / r['total_chunks'] * 100 if r['total_chunks'] > 0 else 0
            print(f"{r['file'][:58]:<60} {r['total_chunks']:>8} {accuracy:>7.1f}%")

        return all_results


if __name__ == "__main__":
    validator = PageValidator(
        data_dir="/home/coder/project/data_job/data/laws_storage",
        output_dir="/home/coder/project/data_job/output"
    )

    results = validator.run_validation()
