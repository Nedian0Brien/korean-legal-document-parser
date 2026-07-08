# Korean Legal Document Parser

한국어 법령·규정 문서의 장-절-조 구조와 페이지 마커를 보존해 구조화하는 문서 파서입니다.

## 책임 범위

- `제1장`, `제1절`, `제1조`, `부칙` 헤더 탐지
- 원본 헤더 문자열 보존
- 페이지 마커 `{1}---` 형식 탐지
- 문서 트리와 hierarchy path 생성
- 구조화 Markdown과 청크 검증용 산출물 생성

Small-to-Big RAG 청킹과 Qdrant ingestion/retrieval은 별도 저장소인 `small-to-big-rag-chunker`에서 관리합니다.

## 구조

```text
src/chunking/
└── LegalDocumentProcessor.py
docs/
└── LegalDocumentProcessor.md
```

## 준비

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 검증

```bash
python -m py_compile src/chunking/LegalDocumentProcessor.py
```

기존 검증 스크립트 중 일부는 원본 개발 환경의 절대 경로를 포함합니다. repo 안정화 단계에서는 `tests/fixtures/` 기반 테스트로 교체해야 합니다.
