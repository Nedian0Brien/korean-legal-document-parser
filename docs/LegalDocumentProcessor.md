# LegalDocumentProcessor
## 개요
- 장-절-조 구조로 구성된 한국어 법령 문서들의 구조를 이해하고 이에 맞추어 파싱 및 청킹을 수행하기 위한 문서 처리 아키텍처

## 구조

### 1. 데이터 클래스 및 보조 클래스
- _HeaderPattern
정규식 패턴, 타입, 계층 순위(rank) 정보를 담는 구조체. _LegalStyleDetector가 내부적으로 사용
- _LegalStyleDetector
법령, 규정 문서에서 헤더(예: 제1장, 제2절, 제3조, 부칙 등)를 탐지하는 전용 객체
    - detect_header(line) -> (rank, text, prefix, type) 반환
    - is_list_type(type_name)은 현재 항상 False (법령 문서에 리스트가 없다고 가정)
- _None
문서 트리 구조를 나타내는 노드 객체
주요 속성: text, level, rank, prefix, type, children, content_line, parent, source_file, page_number 등
* formatted() - 완전한 헤더 문자열 반환
* get_full_content() - 헤더 + 직속 내용 반환
* get_aggregated_content() - 하위 전체 내용까지 포함한 문자열 반환
* get_hierarchy_path() - 루트 -> 현재 노드까지의 헤더 경로 리스트 반환
* pretty_print() / get_pretty_print_str() - 트리를 콘솔 / 문자열 형태로 가시화

### 2. LegalDocumentParser - LlamaIndex 연동 파서
LlamaIndex의 NodeParser 상속, LlamaIndex 구조와의 연동 및 호환성 유지에 유의할 것
- _init_(self, config, category, file_type, text_splitter)
파서 초기화-설정(LegalDocumentProcessor), 카테고리, 파일유형 메타정보, 텍스트 분할기(RecursiveCharacterTextSpiltter)를 보관
- _parse_nodes(self, documents, show_progress=True)
전체 파싱 엔트리. 각 Document를 _parse_single_document로 넘겨 모든 TextNode를 수집 후 반환
- _parse_single_document(self, doc)
단일 문서에 대해 전처리->트리 빌드->청크 생성->후처리->TextNode 변환 순서로 진행하고 (tree, node_list) 반환
- _preprocess_lines(self, lines)
원문 라인들을 읽어 헤더와 본문을 명확히 분리하고, 헤더 라인 뒤에 남은 텍스트를 별도 라인으로 삽입
- _build_tree(self, lines, source_file)
전처리된 라인 리스트를 순차 탐색하며 _detector.detect_header로 헤더를 찾고, 계층(rank)에 맞춰 _Node 트리를 구성
- _find_parent_with_context(self, stack, new_rank, new_type) 
현재 스택(stack)에서 새 노드가 삽입될 적절한 노드를 찾는 로직(리스트 타입 여부, 계층 순위에 따라 판단)
- _tree_to_chunks(self, node)
루트 트리에서 preamble(서문)을 별도 청크로 처리하고, 하위 노드들을 _chunk_and_merge_children으로 재귀 청크화
- _chunk_and_merge_children(self, nodes) 
동일 레벨 형제 노드들을 크기 제한(target_chunks_size, max_chunk_size)에 따라 그룹화/청킹하고, 필요 시 재귀적으로 하위 청크 생성
- _process_child_group(self, group)
그룹(복수 노드) 혹은 단일 노드에 대해 크기 판단
1. 전체가 너무 크면 재분할
2. 단일 노드이며 너무 큰 노드이면 헤더/본문 구분 후 청크 반환
- _create_grouped_chunk(self, nodes)
여러 형제 노드를 하나의 청크로 합칠 때 사용. 전체 텍스트와 계층 경로, 그룹 타이틀을 메타데이터에 담는다.
- _create_node_chunk(self, node, chunk_type, include_children=True)
단일 노드(또는 헤더만)를 청크화. include_children=False이면 헤더만 포함
- _create_chunk_dict(self, **kwargs)
청크 정보를 딕셔너리 형태로 정규화. 자동으로 size, chunk_id(uuid)를 삽입
- _postprocess_chunks(self, chunks)
작은 청크(min_abs_chunk_size보다 작음)를 앞뒤 청크와 병합해 최소 크기를 충족하도록 보정
- _create_chunk_dict(self, chunk_data, doc_metadata)
최종 TextNode 객체 생성. 메타데이터에 category, file_type, hierarchy, chunk_type, chunk_size, detected_type 등을 삽입하고, LLM/Embedding 메타데이터 제외 키(exclude_*_metadata_keys)를 지정
- tree_to_sturctured_md(self, node) 파싱된 트리를 마크다운 형식으로 시각화. pretty-print와 본문 마크다운 두 부분을 결합해 반환

### 3. LegalDocumentProcessor (DocumentProcessor 상속) - 메인 프로세서

- _init_(self, config)
설정 객체(LegalDocumentProcessorConfig)를 저장하고 Qdrant 클라이언트를 초기화한다. 운영/테스트 모드에 따라서 운영 모드에서는 LlamaIndex Embedding 모델 및 Qdrant 클라이언트를 설정한다.
- _save_outputs(self, tree, doc_nodes, parser, output_base_dir, collection_name, category, source_filename)
파싱 결과를 두 파일로 저장한다. 컬렉션-카테고리 별 하위 디렉토리를 자동 생성한다.({output_dir}/{collection_name}/{category/})
1. *_sturctured.md - 트리 구조와 마크다운 내용
2. *_chunks.json - 각 청크(TextNode)의 ID, 텍스트, 메타데이터 JSON.
- index_nodes(self, nodes, collection_name)
Qdrant Vector Store에 주어진 nodes(TextNode 리스트)를 인덱싱한다. QdrantVectorStore -> StorageContext -> VectorStoreIndex 흐름을 사용한다.
- run(self, tasks)
전체 파이프라인 실행 함수.
1. 테스트/운영 모드에 따라 Qdrant 컬렉션 사전 준비(_setup_qdrant_collections)
2. 각 task(입력 폴더, 카테고리, 컬렉션, 파일형식(보여줄 파일 형식 지정))에 대해:
    - 임시 전처리 디렉토리 생성 -> preprocess_markdown_files 실행
    - 문서 로드 (_load_document_from_source - 외부 구현)
    - 텍스트 스플리터 (RecursiveCharacterTextSplitter) 초기화
    - LegalDocumentParser 인스턴스 생성
    - 문서별 파싱 -> 트리,청크,노드 생성
    - 운영 모드이면 Qdrant 인덱싱(index_nodes)
    - 파일 저장 (_save_outputs)
3. 모든 작업 종료 후 임시 디렉터리 정리 및 (운영 모드라면) 카테고리별 텍스트 디코딩(decode_category_node_text, 외부 구현) 실행

## 전체 흐름 요약 (시퀀스)
1. 환경 초기화 (LegalDocumentProcessor.__init__)
2. 작업 리스트(tasks) 전달 -> run() 진입
3. 각 작업에 대해
    - 임시 전처리 디렉토리 생성 & 마크다운 전처리(preprocess_markdown_files)
    - 전처리된 디렉토리에서 LlamaIndex Document 로드(_load_document_from_source)
    - 텍스트 스플리터 설정 -> 파서(LegalDocumentParser) 초기화
    - 문서 단위 루프:
        1. _parse_single_document -> 트리 + 청크 + 노드 목록
        2. (운영 모드) Qdrant 인덱싱(index_nodes)
        3. 파일 저장(_save_outputs) - 구조 Markdown + 청크 JSON
4. 모든 작업이 끝나면 (운영 모드) 카테고리별 텍스트 디코딩 실행
5. 임시 디렉터리 정리 및 종료 로그 출력
 