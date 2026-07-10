import os
import re
import uuid
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Sequence, Callable, Set
from tqdm import tqdm
from dataclasses import dataclass, field
from korean_legal_document_parser.types import DocumentTask, LegalParserConfig

# --- 의존성 설치 안내 ---
# pip install llama-index qdrant-client llama-index-vector-stores-qdrant llama-index-embeddings-huggingface sentence-transformers langchain-text-splitters transformers

try:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document, Settings, StorageContext
    from llama_index.core.node_parser import NodeParser
    from llama_index.core.schema import TextNode, BaseNode, NodeRelationship, RelatedNodeInfo
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from pydantic import PrivateAttr
    import qdrant_client
    from qdrant_client.models import VectorParams, Distance
    
    # 텍스트 스플리터 및 토크나이저
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import PreTrainedTokenizerFast
except ImportError as e:
    print(f"오류: 필요한 라이브러리가 설치되지 않았습니다. ({e})")
    print("pip install langchain-text-splitters transformers 등을 확인해주세요.")
    exit()

# --- 1. 데이터 클래스 및 보조 클래스 ---

@dataclass
class _HeaderPattern:
    """ 헤더 정규식 패턴 및 계층 정보 """
    pattern: re.Pattern
    type: str
    rank: int

class _LegalStyleDetector:
    """ 법령/규정 헤더 탐지 전용 클래스 (엄격한 패턴 적용) """
    def __init__(self):
        self.pattern_definitions = [
            {"full": r'^(?:[#\s]*)(부칙)\b\s*(.*)', "type": "addendum", "rank": 0},
            # [유지] 제1장 + 공백 + 제목 (예: "제1장 총칙")
            {"full": r'^\s*(제\s*\d+\s*장)\s+(.*)', "type": "chapter", "rank": 1},
            # [유지] 제1절 + 공백 + 제목
            {"full": r'^\s*(제\s*\d+\s*절)\s+(.*)', "type": "section", "rank": 2},
            # [유지] 제1조 + (제목) 형태만 허용 (예: "제1조(목적)")
            {"full": r'^\s*(제\s*\d+조(?:의\d+)?)\s*(\([^)]+\))', "type": "article", "rank": 3}
        ]
        self.outline_patterns = [_HeaderPattern(re.compile(p["full"]), p["type"], p["rank"]) for p in self.pattern_definitions]
        # 페이지 마커 정규식 (예: {1}---)
        self.page_marker_pattern = re.compile(r'\{(\d+)\}[-]+\n?')

    def detect_header(self, line: str) -> Optional[Tuple[int, str, str, str, int]]:
        """
        헤더를 감지하고 정보를 반환합니다.

        Returns:
            (rank, text_content, prefix_content, type, match_end_pos)
            match_end_pos: 원본 line에서 매칭된 헤더의 끝 위치 (인덱스)
        """
        stripped = line.strip()
        if not stripped: return None

        # 마크다운 헤더(#, ##, ###, #### 등) 제거
        stripped = re.sub(r'^#{1,6}\s+', '', stripped)
        # 강조 구문(*, _, ~) 제거
        stripped = re.sub(r'^[*_~]+|[*_~]+$', '', stripped).strip()
        # 앞에 대시(- )가 붙는 경우 제거
        stripped = re.sub(r'^-\s+', '', stripped)

        for p in self.outline_patterns:
            match = p.pattern.match(stripped)
            if match:
                groups = match.groups()
                prefix_content = (groups[0] or "").strip()
                text_content = (groups[1] or "").strip()

                # 매칭된 헤더의 끝 위치 (match.end()는 stripped 기준)
                match_end_pos = match.end()

                return p.rank, text_content, prefix_content, p.type, match_end_pos
        return None
    
    def detect_page_marker(self, line: str) -> Optional[int]:
        """ 라인이 페이지 마커인지 확인하고 페이지 번호 반환 """
        match = self.page_marker_pattern.match(line.strip())
        if match:
            return int(match.group(1))
        return None

class _Node:
    """ 문서 계층 구조 트리 노드 """
    def __init__(self, text: str, level: int, rank: int, **kwargs):
        self.text = text          # 예: "총칙" (제목만, detect_header에서 추출된 값)
        self.level = level
        self.rank = rank
        self.prefix = kwargs.get("prefix", "") # 예: "제1장"
        self.type = kwargs.get("detected_type", "unknown")

        # [핵심] 원본 헤더 텍스트 저장 (공백 포함된 원문 그대로)
        # 예: "제1장 총칙" 또는 "제1장  총칙" (원문의 공백 그대로)
        self.original_header = kwargs.get("original_header", f"{self.prefix} {self.text}").strip()

        self.children: List['_Node'] = []
        # (텍스트, 페이지번호) 튜플 리스트로 각 라인의 페이지 정보 보존
        self.content_lines: List[Tuple[str, int]] = []
        self.parent: Optional['_Node'] = None
        self.source_file = kwargs.get("source_file", "")
        # 헤더 페이지: 헤더가 실제로 위치한 페이지
        self.header_page = kwargs.get("header_page", 0)
        # 노드 페이지: content_lines 기반으로 계산됨 (콘텐츠 중간 라인 기준)
        self.page_number = 0

    def calculate_page_number(self) -> None:
        """content_lines의 중간 라인 페이지 번호로 page_number를 설정"""
        if self.content_lines:
            mid_idx = len(self.content_lines) // 2
            self.page_number = self.content_lines[mid_idx][1]

    def get_full_content(self) -> str:
        """
        [노드 텍스트 생성용]
        저장해둔 원본 헤더(original_header)를 그대로 사용하여 원문 훼손을 방지합니다.
        """
        header = self.original_header
        # content_lines가 (텍스트, 페이지) 튜플이므로 텍스트만 추출
        content = "\n".join(line for line, _ in self.content_lines)

        if header:
            return f"{header}\n{content}".strip()
        return content.strip()

    def get_aggregated_content(self) -> str:
        # 자식 노드들의 텍스트를 합칠 때도 get_full_content()를 사용하므로 원본 유지됨
        content_parts = [self.get_full_content()]
        for child in self.children:
            child_content = child.get_aggregated_content()
            if child_content.strip():
                content_parts.append(child_content)
        return "\n\n".join(content_parts)

    def get_hierarchy_path(self) -> List[str]:
        # 원본 헤더를 사용하여 hierarchy 경로 생성
        path, current = [], self
        while current and current.parent and current.parent.text != "ROOT":
            path.insert(0, current.original_header)
            current = current.parent
        return path
    
    def get_pretty_print_str(self, indent: int = 0) -> str:
        # 타입별 이모지 및 표현 정의
        type_icons = {
            'root': '📚',
            'addendum': '📌',
            'chapter': '📖',
            'section': '📄',
            'article': '📋',
            'unknown': '❓'
        }

        icon = type_icons.get(self.type, '•')
        # 헤더 페이지와 노드 페이지가 다르면 둘 다 표시
        if self.header_page > 0 and self.page_number > 0 and self.header_page != self.page_number:
            page_info = f" 〔h.{self.header_page} → p.{self.page_number}〕"
        elif self.page_number > 0:
            page_info = f" 〔p.{self.page_number}〕"
        elif self.header_page > 0:
            page_info = f" 〔h.{self.header_page}〕"
        else:
            page_info = ""

        # 들여쓰기: 계층에 따라 ├─ 또는 │  사용
        if indent == 0:
            prefix = ""
        else:
            prefix = "│  " * (indent - 1) + "├─ "

        # 원본 헤더 사용 (ROOT는 예외)
        formatted_text = self.original_header if self.text != "ROOT" else "문서 전체"
        res = f"{prefix}{icon} {formatted_text}{page_info}\n"

        for child in self.children:
            res += child.get_pretty_print_str(indent + 1)
        return res

# --- 2. LegalDocumentParser (Refactored for S2B & Bottom-up) ---

class RegalDocumentParser(NodeParser):
    # Pydantic PrivateAttr로 private 속성 선언 (언더스코어로 시작해야 함)
    _config: LegalParserConfig = PrivateAttr()
    _category: str = PrivateAttr()
    _file_type: str = PrivateAttr()
    _detector: _LegalStyleDetector = PrivateAttr()
    _tokenizer: PreTrainedTokenizerFast = PrivateAttr()
    _token_len_func: Callable = PrivateAttr()
    _leaf_splitter: RecursiveCharacterTextSplitter = PrivateAttr()
    _parent_splitter: RecursiveCharacterTextSplitter = PrivateAttr()

    def __init__(self, config: LegalParserConfig, category: str, file_type: str):
        super().__init__()
        self._config = config
        self._category = category
        self._file_type = file_type
        self._detector = _LegalStyleDetector()

        # [변경] PreTrainedTokenizerFast 사용
        try:
            if os.path.exists(config.TOKENIZER_FILE_PATH):
                self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.TOKENIZER_FILE_PATH)
            else:
                print(f"⚠️ Warning: 토크나이저 파일({config.TOKENIZER_FILE_PATH})이 없어 기본값(bert-base-multilingual-cased)을 사용합니다.")
                self._tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        except Exception as e:
            print(f"⚠️ Error initializing tokenizer: {e}")
            print("기본 tokenizer를 로드합니다.")
            self._tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-multilingual-cased")

        # [변경] 토큰 길이 계산 함수
        def _token_len(text: str) -> int:
            return len(self._tokenizer.encode(text))

        self._token_len_func = _token_len

        self._leaf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.LEAF_TOKEN_SIZE,
            chunk_overlap=config.LEAF_TOKEN_OVERLAP,
            length_function=_token_len,
            separators=[
                "\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",  # 마크다운 헤더
                "\n\n",  # 이중 개행 (우선순위 높음)
                "\n제",  # 법령 조문 시작 (조 헤더와 본문 분리 방지)
                "\n",  # 단일 개행
                " ",  # 공백
                ""  # 최후의 수단
            ],
            keep_separator=True,  # separator를 다음 청크에 유지하여 조 헤더 보존
            add_start_index=True
        )

        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.PARENT_MAX_TOKEN_SIZE,
            chunk_overlap=int(config.PARENT_MAX_TOKEN_SIZE * 0.1),
            length_function=_token_len,
            separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ", "\n\n", "\n", " ", ""],
            add_start_index=True
        )

    def _get_page_from_offset(self, char_offset: int, char_page_map: List[Tuple[int, int]]) -> int:
        """ character offset에 해당하는 페이지 번호 반환 (이진 탐색) """
        if not char_page_map:
            return 0

        import bisect
        # char_page_map은 [(start_offset, page), ...] 형태로 오름차순 정렬되어 있음
        # char_offset보다 큰 첫 번째 요소의 인덱스를 반환 (bisect_right)

        # 튜플 비교: (offset, MAX_PAGE)로 검색
        idx = bisect.bisect_right(char_page_map, (char_offset, 999999))

        if idx == 0:
            return char_page_map[0][1]

        # idx-1이 char_offset이 포함된 구간의 시작점
        return char_page_map[idx-1][1]

    def _collect_all_lines_recursive(self, node: _Node) -> List[Tuple[str, int]]:
        """
        노드와 그 자식들의 모든 텍스트 라인을 계층 구조 순서대로 (텍스트, 페이지) 리스트로 반환
        get_aggregated_content()가 텍스트를 합치는 순서와 100% 일치해야 함
        """
        lines = []

        # 1. 헤더 (자신의 original_header 텍스트)
        # 헤더 페이지는 node.header_page 사용
        header_text = node.original_header
        if header_text:
            lines.append((header_text, node.header_page))

        # 2. 자신의 본문 내용
        lines.extend(node.content_lines)

        # 3. 자식 노드들의 내용 (재귀)
        for child in node.children:
            lines.extend(self._collect_all_lines_recursive(child))

        return lines

    def _build_content_page_map_direct(self, lines_with_page: List[Tuple[str, int]], separator: str = "\n") -> Tuple[str, List[Tuple[int, int]]]:
        """
        라인 리스트를 결합하여 combined_content를 만들고, 동시에 char_offset -> page 매핑을 생성함.

        Returns:
            (combined_content, char_page_map)
            char_page_map은 [(start_offset, page), ...] 형태
        """
        combined_parts = []
        char_page_map = []
        current_offset = 0
        sep_len = len(separator)

        for text, page in lines_with_page:
            if not text: continue

            # 매핑 기록: 현재 오프셋 지점의 페이지는 'page'임
            char_page_map.append((current_offset, page))

            combined_parts.append(text)
            current_offset += len(text) + sep_len # 텍스트 길이 + 구분자 길이

        combined_content = separator.join(combined_parts)
        return combined_content, char_page_map

    def _parse_nodes(self, documents: Sequence[Document], show_progress: bool = False, **kwargs) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        doc_iterator = tqdm(documents, desc="문서 파싱 중") if show_progress else documents
        
        for doc in doc_iterator:
            _, nodes = self._parse_single_document(doc)
            all_nodes.extend(nodes)
        return all_nodes

    def _parse_single_document(self, doc: Document) -> Tuple[_Node, List[TextNode]]:
        # metadata에서 원본 파일명 추출 (file_name, source, file_path 순서로 시도)
        source_file = Path(
            doc.metadata.get("file_name") or
            doc.metadata.get("source") or
            doc.metadata.get("file_path", "Unknown")
        ).name
        lines = doc.get_content().splitlines()
        
        # 1. 트리 빌드 (페이지 정보 포함)
        processed_lines_with_page = self._preprocess_lines(lines)
        tree = self._build_tree(processed_lines_with_page, source_file)
        
        # 2. Bottom-up 노드 생성
        doc_nodes = self._generate_hierarchical_nodes(tree, doc.metadata)
        return tree, doc_nodes

    def _generate_hierarchical_nodes(self, tree: _Node, doc_metadata: Dict) -> List[TextNode]:
        all_nodes = []

        if tree.text == "ROOT" and tree.content_lines:
            preamble_lines = tree.content_lines
            combined_preamble, preamble_map = self._build_content_page_map_direct(preamble_lines, "\n")

            if combined_preamble.strip():
                # 서문 처리 - 직접 매핑 방식 사용
                if self._token_len_func(combined_preamble) > self._config.PARENT_MAX_TOKEN_SIZE:
                    parent_docs = self._parent_splitter.create_documents([combined_preamble])
                    for i, doc in enumerate(parent_docs):
                        seg = doc.page_content
                        seg_start = doc.metadata.get('start_index', 0)
                        self._create_parent_and_leaves(
                            content=seg, hierarchy_str="서문", range_info=f"(Part {i+1})",
                            doc_metadata=doc_metadata, detected_type="preamble", all_nodes=all_nodes,
                            chunk_type_suffix=f"_preamble_seg{i}",
                            precalc_map=preamble_map, global_offset=seg_start
                        )
                else:
                    self._create_parent_and_leaves(
                        content=combined_preamble, hierarchy_str="서문", range_info="",
                        doc_metadata=doc_metadata, detected_type="preamble", all_nodes=all_nodes,
                        chunk_type_suffix="_preamble",
                        precalc_map=preamble_map, global_offset=0
                    )

        self._traverse_and_group(tree.children, doc_metadata, all_nodes)

        leaf_nodes = [n for n in all_nodes if n.metadata.get("chunk_type") == "leaf"]
        self._establish_sibling_relationships(leaf_nodes)

        return all_nodes

    def _traverse_and_group(self, nodes: List[_Node], doc_metadata: Dict, all_nodes: List[TextNode]):
        node_buffer: List[_Node] = []
        current_buffer_tokens = 0
        
        for node in nodes:
            is_content_node = (node.type == 'article' or not node.children)
            
            if is_content_node:
                node_content = node.get_aggregated_content()
                node_tokens = self._token_len_func(node_content)
                
                if current_buffer_tokens > 0 and (current_buffer_tokens + node_tokens > self._config.PARENT_TARGET_TOKEN_SIZE):
                    self._flush_buffer(node_buffer, doc_metadata, all_nodes)
                    node_buffer = []
                    current_buffer_tokens = 0
                
                node_buffer.append(node)
                current_buffer_tokens += node_tokens
            else:
                if node_buffer:
                    self._flush_buffer(node_buffer, doc_metadata, all_nodes)
                    node_buffer = []
                    current_buffer_tokens = 0
                
                self._traverse_and_group(node.children, doc_metadata, all_nodes)
        
        if node_buffer:
            self._flush_buffer(node_buffer, doc_metadata, all_nodes)

    def _flush_buffer(self, buffer: List[_Node], doc_metadata: Dict, all_nodes: List[TextNode]):
        if not buffer: return

        first_node = buffer[0]

        # 계층 경로 생성 (원본 헤더 사용)
        base_hierarchy = []
        current = first_node
        while current and current.parent and current.parent.text != "ROOT":
            current = current.parent
            base_hierarchy.insert(0, current.original_header)
        hierarchy_str = " > ".join(base_hierarchy)

        # 범위 정보 생성 (원본 헤더 사용)
        titles = [n.original_header for n in buffer]
        range_info = f"Grouped: {' | '.join(titles)}" if len(titles) > 1 else titles[0]

        # [핵심 수정] 버퍼 내 모든 노드(및 자식)의 라인을 재귀적으로 수집
        all_lines_flat: List[Tuple[str, int]] = []
        for node in buffer:
            all_lines_flat.extend(self._collect_all_lines_recursive(node))

        # 직접 매핑 방식으로 content와 매핑 동시 생성
        combined_content, char_page_map = self._build_content_page_map_direct(all_lines_flat, separator="\n")

        total_tokens = self._token_len_func(combined_content)

        if total_tokens > self._config.PARENT_MAX_TOKEN_SIZE:
            # Parent 분할 (Parent Splitter 사용)
            parent_docs = self._parent_splitter.create_documents([combined_content])

            for idx, doc in enumerate(parent_docs):
                segment = doc.page_content
                segment_start = doc.metadata.get('start_index', 0)

                self._create_parent_and_leaves(
                    content=segment,
                    hierarchy_str=hierarchy_str,
                    range_info=f"{range_info} (Part {idx+1})",
                    doc_metadata=doc_metadata,
                    detected_type=first_node.type,
                    all_nodes=all_nodes,
                    chunk_type_suffix=f"_seg_{idx+1}",
                    precalc_map=char_page_map,
                    global_offset=segment_start
                )
        else:
            self._create_parent_and_leaves(
                content=combined_content,
                hierarchy_str=hierarchy_str,
                range_info=range_info,
                doc_metadata=doc_metadata,
                detected_type=first_node.type,
                all_nodes=all_nodes,
                precalc_map=char_page_map,
                global_offset=0
            )

    def _create_parent_and_leaves(self, content: str, hierarchy_str: str, range_info: str,
                                  doc_metadata: Dict, detected_type: str, all_nodes: List[TextNode],
                                  chunk_type_suffix: str = "",
                                  precalc_map: List[Tuple[int, int]] = None,
                                  global_offset: int = 0,
                                  lines_with_page: List[Tuple[str, int]] = None):

        full_hierarchy_header = f"{hierarchy_str} > {range_info}".strip(" > ")

        # 1. Parent Node
        parent_node = TextNode(
            text=content,  # hierarchy 제외, 순수 content만 저장
            id_=str(uuid.uuid4()),
            metadata={
                **doc_metadata,
                "category": self._category,
                "filetype": self._file_type,
                "hierarchy": full_hierarchy_header,
                "chunk_type": f"parent{chunk_type_suffix}",
                "chunk_size": self._token_len_func(content),
                "detected_type": detected_type,
                "page": "0"
            },
            excluded_embed_metadata_keys = ["source", "chunk_type", "chunk_size", "page"],
            excluded_llm_metadata_keys = ["source", "chunk_type", "chunk_size"]
        )
        all_nodes.append(parent_node)

        # 2. Leaf Nodes
        leaf_docs = self._leaf_splitter.create_documents([content])
        leaf_pages: List[int] = []

        # 현재 활성 헤더 추적 로직 (기존 유지)
        current_active_header = range_info.split(" | ")[0] if " | " in range_info else range_info
        if current_active_header.startswith("Grouped: "):
            current_active_header = current_active_header[9:]

        for leaf_doc in leaf_docs:
            leaf_text = leaf_doc.page_content
            # leaf_start는 content 내의 시작점
            leaf_start_local = leaf_doc.metadata.get('start_index', 0)

            # 전역 오프셋으로 변환 (combined_content 기준)
            leaf_abs_start = global_offset + leaf_start_local

            # 미리 계산된 맵에서 페이지 조회
            if precalc_map:
                leaf_page = self._get_page_from_offset(leaf_abs_start, precalc_map)
            else:
                leaf_page = 0  # Fallback

            leaf_pages.append(leaf_page)

            # --- 헤더 탐지 로직 (기존 유지) ---
            found_headers_in_leaf = []
            for line in leaf_text.splitlines():
                h_info = self._detector.detect_header(line)
                if h_info:
                    rank, text, prefix, dtype, match_end = h_info
                    # 원본 라인에서 헤더 부분만 추출 (매칭 끝 위치까지)
                    full_h = line[:match_end].strip()
                    found_headers_in_leaf.append(full_h)

            if found_headers_in_leaf:
                leaf_range = f"Grouped: {' | '.join(found_headers_in_leaf)}" if len(found_headers_in_leaf) > 1 else found_headers_in_leaf[0]
                current_active_header = found_headers_in_leaf[-1]
            else:
                leaf_range = current_active_header

            leaf_full_hierarchy = f"{hierarchy_str} > {leaf_range}".strip(" > ")

            leaf_node = TextNode(
                text=leaf_text,  # hierarchy 제외, 순수 content만 저장
                id_=str(uuid.uuid4()),
                metadata={
                    **doc_metadata,
                    "category": self._category,
                    "filetype": self._file_type,
                    "hierarchy": leaf_full_hierarchy,
                    "chunk_type": "leaf",
                    "chunk_size": self._token_len_func(leaf_text),
                    "detected_type": detected_type,
                    "page": str(leaf_page)  # 정확한 페이지 할당
                },
                excluded_embed_metadata_keys = ["source", "chunk_type", "chunk_size", "page"],
                excluded_llm_metadata_keys = ["source", "chunk_type", "chunk_size"]
            )
            # 관계 설정
            leaf_node.relationships[NodeRelationship.PARENT] = parent_node.as_related_node_info()
            parent_node.relationships.setdefault(NodeRelationship.CHILD, []).append(leaf_node.as_related_node_info())
            all_nodes.append(leaf_node)

        # 3. Parent 페이지 업데이트
        if leaf_pages:
            # 중간값 사용
            mid_idx = len(leaf_pages) // 2
            parent_node.metadata["page"] = str(leaf_pages[mid_idx])

    def _establish_sibling_relationships(self, nodes: List[TextNode]):
        for i, node in enumerate(nodes):
            if i > 0:
                node.relationships[NodeRelationship.PREVIOUS] = nodes[i - 1].as_related_node_info()
            if i < len(nodes) - 1:
                node.relationships[NodeRelationship.NEXT] = nodes[i + 1].as_related_node_info()

    # --- 전처리 및 트리 빌드 (페이지 인식 포함) ---
    def _preprocess_lines(self, lines: List[str]) -> List[Tuple[str, int]]:
        processed = []
        current_page = 1  # 기본값을 1로 설정 (첫 페이지 마커 전까지는 1페이지)

        for line in lines:
            line = line.strip()
            if not line: continue

            # 페이지 마커 처리
            page_num = self._detector.detect_page_marker(line)
            if page_num is not None:
                # 페이지 마커가 0-based이므로 +1하여 1-based로 변환
                # {0}--- → 1페이지, {1}--- → 2페이지
                current_page = page_num + 1
                continue

            # 1. 마크다운(#)만 제거한 "순수 원본 라인" 확보
            # 예: "## 제1장 총칙"  -> "제1장 총칙"
            # 예: "## 제1조(목적) 이 법의 목적은..." -> "제1조(목적) 이 법의 목적은..."
            line_without_md = re.sub(r'^#{1,6}\s+', '', line).strip()

            # 2. 헤더 감지 수행 (line_without_md 기준)
            header_info = self._detector.detect_header(line_without_md)

            if header_info:
                rank, text, prefix, dtype, match_end = header_info

                # [핵심] 원본 라인에서 헤더 부분과 나머지 분리
                # match_end는 stripped된 line_without_md 기준이므로 그대로 사용
                header_original = line_without_md[:match_end].strip()
                remaining = line_without_md[match_end:].strip()

                # 헤더는 헤더로 추가
                processed.append((header_original, current_page))

                # 나머지가 있으면 본문으로 추가
                if remaining:
                    processed.append((remaining, current_page))

            else:
                # 헤더가 아니면 일반 텍스트로 처리
                processed.append((line, current_page))

        return processed

    def _build_tree(self, lines_with_page: List[Tuple[str, int]], source_file: str) -> _Node:
        root = _Node("ROOT", level=0, rank=0, source_file=source_file, detected_type='root', original_header="")
        stack = [root]

        for line, page_num in lines_with_page:
            # line은 _preprocess_lines에서 넘어온 헤더 또는 본문 라인
            header_info = self._detector.detect_header(line)
            if header_info:
                rank, text, prefix, dtype, match_end = header_info
                parent = self._find_parent_with_context(stack, rank, dtype)

                # [핵심] 원본 헤더를 그대로 저장
                # line은 이미 헤더 부분만 들어옴 (_preprocess_lines에서 분리됨)
                node = _Node(
                    text=text,           # "총칙" 또는 "(목적)"
                    level=parent.level + 1,
                    rank=rank,
                    prefix=prefix,       # "제1장" 또는 "제1조"
                    source_file=source_file,
                    detected_type=dtype,
                    header_page=page_num,
                    original_header=line # 원본 헤더 (예: "제1장 총칙" 또는 "제1조(목적)")
                )
                node.parent = parent
                parent.children.append(node)

                new_stack = []
                curr = node
                while curr:
                    new_stack.insert(0, curr)
                    curr = curr.parent
                stack = new_stack
            else:
                # 콘텐츠 라인을 (텍스트, 페이지) 튜플로 저장
                stack[-1].content_lines.append((line, page_num))

        # 트리 빌드 완료 후 모든 노드의 페이지 번호 계산
        self._calculate_all_page_numbers(root)
        return root

    def _calculate_all_page_numbers(self, node: _Node) -> None:
        """트리의 모든 노드에 대해 페이지 번호를 계산"""
        node.calculate_page_number()
        for child in node.children:
            self._calculate_all_page_numbers(child)

    def _find_parent_with_context(self, stack: List[_Node], new_rank: int, new_type: str) -> _Node:
        for i in range(len(stack) - 1, -1, -1):
            if stack[i].rank < new_rank:
                return stack[i]
        return stack[0]

    def tree_to_structured_md(self, node: _Node, doc_nodes: List[TextNode] = None) -> str:
        md = f"# 📄 문서 구조 분석: {node.source_file}\n\n"

        # 1. 계층 트리 시각화
        md += "## 1. 📊 계층 트리 (가시화)\n```text\n"
        md += node.get_pretty_print_str()
        md += "```\n\n---\n\n"

        # 2. 원본 문서 구조 인터랙티브 뷰 (HTML)
        md += "## 2. 📖 원본 문서 구조 (인터랙티브)\n\n"
        md += "_클릭하여 접기/펼치기_\n\n"
        md += self._generate_interactive_html(node)
        md += "\n\n---\n\n"

        # 3. Small-to-Big 청킹 결과 (Parent-Leaf 관계)
        if doc_nodes:
            md += "## 3. 🔗 Small-to-Big 청킹 결과 (Parent-Leaf 관계)\n\n"
            md += "_Small-to-Big 전략에 따라 생성된 Parent와 Leaf 노드의 관계를 확인할 수 있습니다._\n\n"
            md += self._generate_parent_leaf_html(doc_nodes)
        else:
            md += "## 3. 📝 전체 내용 (마크다운)\n\n"
            md += node.get_aggregated_content()

        return md

    def _generate_interactive_html(self, root: _Node) -> str:
        """
        HTML <details> 태그를 사용하여 접었다 폈다 할 수 있는
        계층 구조 뷰를 생성합니다.
        """
        html_lines = []

        def _get_snippet(text: str, length: int = 60) -> str:
            """텍스트 스니펫 생성"""
            clean_text = text.replace("\n", " ").strip()
            return (clean_text[:length] + "...") if len(clean_text) > length else clean_text

        def _traverse_node(node: _Node, level: int = 0):
            """노드를 재귀적으로 순회하며 HTML 생성"""
            if node.text == "ROOT":
                # 루트 노드는 자식들만 처리
                for child in node.children:
                    _traverse_node(child, level)
                return

            # 타입별 이모지
            type_icons = {
                'addendum': '📌',
                'chapter': '📖',
                'section': '📄',
                'article': '📋',
                'unknown': '❓'
            }
            icon = type_icons.get(node.type, '📄')

            # 페이지 정보
            page_info = f" 〔p.{node.page_number}〕" if node.page_number > 0 else ""

            # 스니펫 생성
            content = node.get_full_content()
            snippet = _get_snippet(content, 80)

            # 타입별 스타일
            if node.type == 'addendum':
                border_color = "#ffc107"
                bg_color = "#fff9e6"
            elif node.type == 'chapter':
                border_color = "#28a745"
                bg_color = "#e8f5e9"
            elif node.type == 'section':
                border_color = "#17a2b8"
                bg_color = "#e3f2fd"
            elif node.type == 'article':
                border_color = "#6c757d"
                bg_color = "#f8f9fa"
            else:
                border_color = "#dee2e6"
                bg_color = "#ffffff"

            # 자식 노드 여부 확인
            has_children = len(node.children) > 0

            # 레벨 0는 기본적으로 열림
            open_attr = "open" if level == 0 else ""

            # Details 태그 시작
            html_lines.append(
                f"<details {open_attr} style='margin-left: {level*20}px; "
                f"border-left: 3px solid {border_color}; "
                f"background-color: {bg_color}; "
                f"padding: 8px; margin-bottom: 8px; border-radius: 4px;'>"
            )

            # Summary (제목) - 원본 헤더 사용
            formatted_title = node.original_header
            html_lines.append(
                f"<summary style='cursor: pointer; font-weight: bold; color: #333;'>"
                f"{icon} <strong>{formatted_title}</strong>{page_info} "
                f"<span style='color: #666; font-size: 0.9em; font-weight: normal;'>"
                f"| {node.type.upper()}</span>"
                f"</summary>"
            )

            # 본문 내용
            if node.content_lines:
                html_lines.append(
                    "<div style='margin-top: 10px; padding: 10px; "
                    "background-color: white; border-radius: 3px; "
                    "font-family: monospace; font-size: 0.9em; color: #000; "
                    "white-space: pre-wrap; border: 1px solid #e0e0e0;'>"
                )
                # content_lines가 (텍스트, 페이지) 튜플이므로 텍스트만 추출
                content_text = "\n".join(line for line, _ in node.content_lines)
                html_lines.append(content_text)
                html_lines.append("</div>")

            # 자식 노드 처리
            if has_children:
                html_lines.append("<div style='margin-top: 10px;'>")
                for child in node.children:
                    _traverse_node(child, level + 1)
                html_lines.append("</div>")

            # Details 태그 종료
            html_lines.append("</details>")

        # 트리 순회 시작
        _traverse_node(root)

        return "\n".join(html_lines)

    def _generate_parent_leaf_html(self, doc_nodes: List[TextNode]) -> str:
        """
        Small-to-Big 청킹 결과 (Parent-Leaf 관계)를
        HTML <details> 태그로 표현합니다.
        """
        html_lines = []

        # Parent 노드만 필터링 (chunk_type이 'parent'로 시작하는 것들)
        parent_nodes = [n for n in doc_nodes if n.metadata.get("chunk_type", "").startswith("parent")]

        # 노드 ID로 매핑
        node_map = {n.node_id: n for n in doc_nodes}

        def _get_snippet(text: str, length: int = 60) -> str:
            """텍스트 스니펫 생성"""
            clean_text = text.replace("\n", " ").strip()
            return (clean_text[:length] + "...") if len(clean_text) > length else clean_text

        def _format_hierarchy_with_br(hierarchy_text: str, parent_id: str = "", display_mode: str = "inline", leaf_mapping: dict = None, article_titles: dict = None) -> str:
            """
            계층 구조 텍스트 가독성 개선
            display_mode: 'grid' (Parent용 2열 버튼), 'inline' (Leaf용 가로 라벨)
            leaf_mapping: {조번호: leaf_index} (색상 구분용)
            article_titles: {조번호: 제목} (라벨 옆 제목 표시용)
            """
            import re
            formatted = hierarchy_text
            leaf_mapping = leaf_mapping or {}
            article_titles = article_titles or {}
            
            prefix_html = ""
            articles_text = ""
            prefix_text = ""
            
            # 1. 텍스트 파싱 (Prefix와 Article 분리)
            if "Grouped: " in formatted:
                # "... > Grouped: 제1조... | 제2조..."
                grouped_match = re.search(r'(.*?)\s*>\s*Grouped:\s*(.*)', formatted)
                if grouped_match:
                    prefix_text = grouped_match.group(1)
                    articles_text = grouped_match.group(2)
                else:
                    # "Grouped: 제1조... | 제2조..." (상위 계층 없이 바로 Grouped 시작)
                    grouped_only_match = re.search(r'^Grouped:\s*(.*)', formatted)
                    if grouped_only_match:
                        prefix_text = ""
                        articles_text = grouped_only_match.group(1)
            else:
                # 단일 조 또는 일반 계층 구조 ("제1장 > 제1절 > 제1조")
                parts = formatted.split(" > ")
                # 마지막 요소가 '제N조' 패틴인지 확인
                last_part = parts[-1]
                if re.search(r'제\s*\d+조', last_part):
                    prefix_text = " > ".join(parts[:-1])
                    articles_text = last_part
                else:
                    prefix_text = formatted
                    articles_text = ""

            # 2. Prefix 스타일링 (Parent 모드에서만 표시)
            if display_mode == "grid" and prefix_text:
                prefix_parts = prefix_text.split(" > ")
                styled_prefix = []
                for i, p_part in enumerate(prefix_parts):
                    if i == len(prefix_parts) - 1:
                        styled_prefix.append(f"<span style='font-size: 1.05em; font-weight: bold; color: #333;'>{p_part}</span>")
                    else:
                        styled_prefix.append(f"<span style='font-size: 0.9em; color: #888;'>{p_part}</span>")
                prefix_html = " <span style='color: #ccc;'>&gt;</span> ".join(styled_prefix)

            # 3. Article 라벨/버튼 생성 (조 번호가 있는 경우)
            if articles_text:
                article_parts = articles_text.split(" | ")
                item_parts = []
                
                for part in article_parts:
                    article_match = re.search(r'(제\s*\d+조(?:의\d+)?)', part)
                    if article_match:
                        article_num = article_match.group(1)
                        
                        # 색상 변수 계산 (Leaf Index 기반)
                        leaf_idx = leaf_mapping.get(article_num, 0)
                        hue_adj = ((leaf_idx % 5) * 25) - 50
                        
                        # 스타일 공통 변수
                        base_hue = 260 + hue_adj
                        
                        if display_mode == "grid":
                            # [Parent] 그라디언트 버튼
                            btn_style = (
                                f"background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
                                f"filter: hue-rotate({hue_adj}deg); "
                                f"color: white; border: none; padding: 4px 8px; margin: 2px 8px 2px 2px; "
                                f"border-radius: 6px; cursor: default; font-weight: bold; "
                                f"font-size: 0.85em; box-shadow: 0 1px 3px rgba(0,0,0,0.15); "
                                f"white-space: nowrap; min-width: 60px; text-align: center; display: inline-block;"
                            )
                            # 제목 추출 (article_num을 제외한 나머지)
                            remaining_text = part.replace(article_num, "").strip()
                            if not remaining_text:
                                html_part = f'<button style="{btn_style}">{article_num}</button> <span style="color: #888;">-</span>'
                            else:
                                html_part = part.replace(article_num, f'<button style="{btn_style}">{article_num}</button>')
                            
                            html_part = f"<div style='display: flex; align-items: center;'>{html_part}</div>"
                        else:
                            # [Leaf] 라벨 + 제목
                            label_style = (
                                f"color: hsl({base_hue}, 70%, 30%); "
                                f"background-color: hsl({base_hue}, 70%, 93%); "
                                f"border: 1px solid hsl({base_hue}, 40%, 70%); "
                                f"padding: 1px 8px; margin: 0 4px 0 0; "
                                f"border-radius: 4px; font-weight: bold; font-size: 0.8em; "
                                f"display: inline-block; white-space: nowrap;"
                            )
                            # 제목 표시 (없으면 -)
                            title_text = article_titles.get(article_num, "").strip()
                            if not title_text:
                                title_text = "-"
                            
                            title_html = f"<span style='font-size: 0.85em; color: #666; font-weight: normal; margin-right: 8px;'>{title_text}</span>"
                                
                            html_part = f'<span style="{label_style}">{article_num}</span>{title_html}'
                            # part 문자열 치환 대신 바로 생성된 HTML 사용 (part엔 제N조 외의 텍스트가 거의 없으므로)
                        
                        item_parts.append(html_part)
                    else:
                        if display_mode == "grid":
                            item_parts.append(f"<div>{part}</div>")
                        else:
                            item_parts.append(part)

                # 4. 레이아웃 조립
                if display_mode == "grid":
                    grid_html = (
                        f"<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; "
                        f"margin-top: 8px; padding: 8px; background-color: rgba(0,0,0,0.02); border-radius: 4px;'>"
                        f"{''.join(item_parts)}</div>"
                    )
                    return f"{prefix_html}<br>{grid_html}"
                else:
                    # Leaf 모드: Prefix 없이 라벨들만 반환
                    flex_html = (
                        f"<div style='display: flex; flex-wrap: wrap; row-gap: 4px; column-gap: 4px; align-items: center;'>"
                        f"{''.join(item_parts)}</div>"
                    )
                    return flex_html
            
            # 조 번호가 없는 일반적인 경우
            if display_mode == "inline":
                return "" # Leaf에선 불필요한 정보 숨김
                
            formatted = formatted.replace(" > ", " <span style='color: #ccc;'>&gt;</span> ")
            return formatted

        # 통계 정보
        total_parents = len(parent_nodes)
        total_leaves = len([n for n in doc_nodes if n.metadata.get("chunk_type") == "leaf"])

        html_lines.append(f"### 📊 청킹 통계\n")
        html_lines.append(f"- **Parent 노드**: {total_parents}개\n")
        html_lines.append(f"- **Leaf 노드**: {total_leaves}개\n")
        html_lines.append(f"- **평균 Leaf per Parent**: {total_leaves / total_parents if total_parents > 0 else 0:.1f}개\n\n")

        # Parent 노드별로 순회
        for idx, parent in enumerate(parent_nodes, 1):
            # ... (중략: 메타데이터, Leaf Mapping 생성 로직은 유지) ...
            
            # Leaf Mapping (이전 단계에서 이미 구현됨)
            leaf_mapping = {}
            import re
            child_leaves = []
            if NodeRelationship.CHILD in parent.relationships:
                 child_info_list = parent.relationships[NodeRelationship.CHILD]
                 if isinstance(child_info_list, list):
                    for child_info in child_info_list:
                        if child_info and child_info.node_id in node_map:
                            child_node = node_map[child_info.node_id]
                            if child_node.metadata.get("chunk_type") == "leaf":
                                child_leaves.append(child_node)
            
            for l_idx, leaf in enumerate(child_leaves, 1):
                l_h = leaf.metadata.get("hierarchy", "")
                articles = re.findall(r'(제\s*\d+조(?:의\d+)?)', l_h)
                for art in articles:
                    leaf_mapping[art] = l_idx
            
            leaf_count = len(child_leaves)
            
            # Parent 메타데이터
            hierarchy = parent.metadata.get("hierarchy", "N/A")
            chunk_size = parent.metadata.get("chunk_size", 0)
            detected_type = parent.metadata.get("detected_type", "unknown")
            page_label = parent.metadata.get("page_label", "N/A")

            # Parent Header
            parent_id = f"parent_{idx}"
            headers_display = _format_hierarchy_with_br(hierarchy, parent_id, display_mode="grid", leaf_mapping=leaf_mapping)
            
            # ... (색상 설정 생략 - 그대로 둠) ...
            if detected_type == 'addendum': border_color = "#ffc107"; bg_color = "#fff9e6"
            elif detected_type == 'chapter': border_color = "#28a745"; bg_color = "#e8f5e9"
            elif detected_type == 'section': border_color = "#17a2b8"; bg_color = "#e3f2fd"
            elif detected_type == 'article': border_color = "#6c757d"; bg_color = "#f8f9fa"
            else: border_color = "#007bff"; bg_color = "#e7f3ff"

            html_lines.append(
                f"<details style='border: 2px solid {border_color}; "
                f"background-color: {bg_color}; padding: 12px; margin-bottom: 15px; border-radius: 6px;'>"
            )
            
            # Parent Summary
            page_info = f"p.{page_label}" if page_label and page_label != "N/A" and page_label != "0" else ""
            html_lines.append(
                f"<summary style='cursor: pointer; font-weight: bold; font-size: 1.05em; color: #333;'>"
                f"🌳 <strong>Parent #{idx}</strong> "
                f"<div style='color: #444; font-size: 0.95em; margin-top: 6px; padding-left: 24px; line-height: 1.5;'>"
                f"↳ {headers_display}"
                f"</div>"
                f"<div style='margin-top: 8px; padding-left: 24px;'>"
                f"<span style='background-color: #17a2b8; color: white; padding: 2px 6px; "
                f"border-radius: 3px; font-size: 0.85em;'>📄 {leaf_count} Leaves</span> "
                f"<span style='background-color: #6c757d; color: white; padding: 2px 6px; "
                f"border-radius: 3px; font-size: 0.85em; margin-left: 4px;'>📏 {chunk_size} tokens</span>"
            )
            if page_info:
                html_lines.append(f" <span style='background-color: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; margin-left: 4px;'>📚 {page_info}</span>")
            html_lines.append("</div></summary>")

            # Leaf 노드들
            if child_leaves:
                html_lines.append("<div style='margin-top: 15px;'>")
                html_lines.append(f"<strong style='color: #17a2b8;'>🍃 Leaf Nodes ({leaf_count}개):</strong>")

                for leaf_idx, leaf in enumerate(child_leaves, 1):
                    leaf_size = leaf.metadata.get("chunk_size", 0)
                    leaf_hierarchy = leaf.metadata.get("hierarchy", "N/A")
                    leaf_snippet = _get_snippet(leaf.get_content(), 120)
                    leaf_page = leaf.metadata.get("page", "0")  # Leaf 노드의 페이지 정보 추출

                    # [추가] Content에서 조 제목 추출 (제N조 -> 제목)
                    # 예: "제37조(청문회)" -> {'제37조': '청문회'}
                    article_titles = {}
                    lines = leaf.get_content().splitlines()
                    for line in lines:
                        # "제37조(청문회)..." 또는 "제37조 ..." 패턴
                        match = re.search(r'^\s*(제\s*\d+조(?:의\d+)?)\s*[:(]?\s*([^)\n]*)', line)
                        if match:
                            art_num = match.group(1).replace(' ', '') # 제37조
                            art_title = match.group(2).strip() # 청문회
                            if art_num and art_title:
                                article_titles[art_num] = art_title

                    # Leaf Header (라벨들) - titles 전달
                    leaf_headers_display = _format_hierarchy_with_br(leaf_hierarchy, f"{parent_id}_leaf_{leaf_idx}", display_mode="inline", leaf_mapping=leaf_mapping, article_titles=article_titles)

                    # 앵커 ID 생성
                    import re
                    anchor_id = ""
                    article_match = re.search(r'(제\s*\d+조(?:의\d+)?)', leaf_hierarchy)
                    if article_match:
                        article_num = article_match.group(1).replace(' ', '')
                        anchor_id = f"{parent_id}_{article_num}"
                    
                    # [동적 배경색 생성]
                    hue_adj = ((leaf_idx % 5) * 25) - 50
                    base_hue = 260 + hue_adj
                    # 아주 연한 파스텔톤 배경
                    leaf_bg_color = f"hsl({base_hue}, 60%, 97%)"
                    leaf_border_color = f"hsl({base_hue}, 40%, 80%)"

                    html_lines.append(
                        f"<details id='{anchor_id}' style='margin-left: 20px; margin-top: 8px; "
                        f"border-left: 4px solid {leaf_border_color}; background-color: {leaf_bg_color}; "
                        f"padding: 8px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>"
                    )
                    
                    # Summary - Leaf 페이지 정보 추가
                    leaf_page_info = ""
                    if leaf_page and leaf_page != "0":
                        leaf_page_info = f"<span style='background-color: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; margin-left: 4px;'>📚 p.{leaf_page}</span> "

                    html_lines.append(
                        f"<summary style='cursor: pointer; font-size: 0.95em; color: #555;'>"
                        f"🍃 <strong>Leaf #{leaf_idx}</strong> "
                        f"<span style='color: #999; font-size: 0.85em;'>| {leaf_size} tokens</span> "
                        f"{leaf_page_info}"
                        f"<div style='margin-top: 6px; margin-bottom: 6px; padding-left: 4px;'>{leaf_headers_display}</div>"
                        f"<div style='margin-top: 4px; padding: 6px 10px; border-left: 2px solid #ddd; "
                        f"background: rgba(255,255,255,0.6); color: #555; font-size: 0.9em; border-radius: 0 4px 4px 0;'>"
                        f"{leaf_snippet}</div>"
                        f"</summary>"
                    )
                    
                    # Details Body: Hierarchy(Breadcrumb) + Header/Content Blocks
                    
                    # 1. Hierarchy Breadcrumb (텍스트 스타일)
                    breadcrumb_html = ""
                    if " > " in leaf_hierarchy:
                        parts = leaf_hierarchy.split(" > ")
                        
                        # 마지막 파트(조/Grouped) 제외하고 상위 경로만 표시
                        last_part = parts[-1]
                        should_exclude = False
                        
                        # "Grouped:" 포함되어 있거나, "제N조" 형식이면 제외
                        if "Grouped:" in last_part or re.match(r'^\s*제\s*\d+조', last_part):
                            should_exclude = True
                            
                        path_parts = parts[:-1] if should_exclude else parts
                        
                        if path_parts:
                            styled_path = []
                            for i, p in enumerate(path_parts):
                                if i == len(path_parts) - 1:
                                    # 상위 경로 중 가장 하위: 진하게
                                    styled_path.append(f"<span style='font-weight: bold; color: #333;'>{p}</span>")
                                else:
                                    # 그 외 상위 경로: 연하게
                                    styled_path.append(f"<span style='color: #888;'>{p}</span>")
                            
                            breadcrumb_html = (
                                f"<div style='font-size: 0.9em; color: #666; margin-bottom: 12px; margin-top: 4px;'>"
                                f"{' <span style=color:#ccc>&gt;</span> '.join(styled_path)}"
                                f"</div>"
                            )
                    
                    # 2. Content 파싱 (다중 조 헤더 처리)
                    raw_content = leaf.get_content()

                    content_blocks = []
                    current_lines = []

                    for line in raw_content.split('\n'):
                        # '제N조'로 시작하는 라인을 헤더로 인식
                        if re.match(r'^\s*제\s*\d+조', line):
                            # 이전에 쌓인 텍스트가 있으면 블록으로 추가
                            if current_lines:
                                content_blocks.append({'type': 'text', 'content': '\n'.join(current_lines)})
                                current_lines = []
                            content_blocks.append({'type': 'header', 'content': line.strip()})
                        else:
                            current_lines.append(line)
                    # 남은 텍스트 추가
                    if current_lines:
                        content_blocks.append({'type': 'text', 'content': '\n'.join(current_lines)})

                    # 3. 렌더링
                    html_lines.append(
                        "<div style='margin-top: 8px; padding: 16px; background-color: white; "
                        "border-radius: 4px; border: 1px solid #eee; font-size: 0.9em; line-height: 1.6;'>"
                    )
                    
                    if breadcrumb_html:
                        html_lines.append(breadcrumb_html)
                    
                    for block in content_blocks:
                        if block['type'] == 'header':
                            # 헤더 스타일: 산세리프, 크고 진하게
                            html_lines.append(
                                f"<div style='font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif; "
                                f"font-size: 1.15em; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #e9ecef; "
                                f"padding-bottom: 8px; margin-top: 24px; margin-bottom: 12px;'>"
                                f"{block['content']}</div>"
                            )
                        else:
                            # 본문 스타일: 고정폭, 줄바꿈 유지
                            html_lines.append(
                                f"<div style='font-family: monospace; color: #333; white-space: pre-wrap;'>"
                                f"{block['content']}</div>"
                            )
                            
                    html_lines.append("</div>")
                    html_lines.append("</details>")
                html_lines.append("</div>")
            else:
                 html_lines.append("<div style='margin-top: 10px; padding: 8px; background-color: #fff3cd; border-left: 3px solid #ffc107; color: #856404;'>⚠️ 이 Parent는 Leaf 노드를 가지고 있지 않습니다.</div>")

            html_lines.append("</details>")

        return "\n".join(html_lines)


# --- 3. LegalDocumentProcessor (메인 프로세서) ---

class LegalDocumentProcessor:
    def __init__(self, config: LegalParserConfig, test_mode: bool = True):
        self.config = config
        self.test_mode = test_mode
        self.qdrant_client = None
        
        if not test_mode:
            print("🔧 운영 모드: 임베딩 및 VectorDB를 초기화합니다.")
            Settings.embed_model = HuggingFaceEmbedding(model_name="jhgan/ko-sroberta-multitask")
            Settings.llm = None
            q_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            self.qdrant_client = qdrant_client.QdrantClient(url=q_url)
        else:
            print("🧪 테스트 모드: 로컬 파일 저장만 수행합니다.")
            Settings.embed_model = None
            Settings.llm = None

    def _setup_qdrant_collections(self, tasks: List[DocumentTask]):
        if self.test_mode or not self.qdrant_client: return
        
        unique_collections = {task.collection_name for task in tasks}
        try:
            vector_size = len(Settings.embed_model.get_text_embedding("test"))
        except:
            vector_size = 768

        print("🔄 컬렉션 초기화 작업을 수행합니다...")
        for col_name in unique_collections:
            if self.qdrant_client.collection_exists(col_name):
                self.qdrant_client.delete_collection(col_name)
                print(f"   🗑️  기존 컬렉션 삭제됨: {col_name}")
            
            self.qdrant_client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"   ✅ 새 컬렉션 생성됨: {col_name}")

    def index_nodes(self, nodes: List[BaseNode], collection_name: str):
        if self.test_mode: return
        
        print(f"🚀 '{collection_name}' 컬렉션에 {len(nodes)}개 노드 업로드 시작...")
        try:
            vector_store = QdrantVectorStore(
                client=self.qdrant_client, 
                collection_name=collection_name,
                enable_hybrid=False, 
                batch_size=64
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
            print(f"✅ 업로드 완료.")
        except Exception as e:
            print(f"❌ 업로드 실패: {e}")

    def _save_outputs(self, tree: _Node, doc_nodes: List[TextNode], parser: RegalDocumentParser,
                      collection_name: str, category: str, source_filename: str):
        target_dir = os.path.join(self.config.OUTPUT_DIR, collection_name, category)
        os.makedirs(target_dir, exist_ok=True)
        base_name = os.path.splitext(source_filename)[0]

        # 마크다운 리포트 생성 (트리 + 원본 구조 + Parent-Leaf 관계)
        md_content = parser.tree_to_structured_md(tree, doc_nodes)
        with open(os.path.join(target_dir, f"{base_name}_structured.md"), "w", encoding="utf-8") as f:
            f.write(md_content)

        json_path = os.path.join(target_dir, f"{base_name}_chunks.json")
        chunks_data = []
        for n in doc_nodes:
            # relationships 처리: 리스트인 경우와 단일 객체인 경우를 구분
            rels = {}
            for k, v in n.relationships.items():
                if v is None:
                    rels[str(k.value)] = None
                elif isinstance(v, list):
                    rels[str(k.value)] = [item.node_id if item else None for item in v]
                else:
                    rels[str(k.value)] = v.node_id

            chunks_data.append({
                "id": n.node_id,
                "text": n.text,
                "metadata": n.metadata,
                "relationships": rels
            })
            
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=4)

    def run(self, tasks: List[DocumentTask]):
        print(f"🚀 총 {len(tasks)}개의 작업을 시작합니다.")
        self._setup_qdrant_collections(tasks)

        for task in tasks:
            print(f"\n[작업 시작] {task.category} ({task.source})")

            try:
                reader = SimpleDirectoryReader(input_dir=task.source, required_exts=[".md"], recursive=task.recursive)
                documents = reader.load_data(show_progress=True)

                if not documents:
                    print(f"⚠️ 처리할 파일이 없습니다: {task.source}")
                    continue

                parser = RegalDocumentParser(self.config, task.category, task.file_type)

                all_task_nodes = []
                for doc in tqdm(documents, desc="문서별 처리 중"):
                    tree, doc_nodes = parser._parse_single_document(doc)
                    all_task_nodes.extend(doc_nodes)

                    # metadata에서 원본 파일명 추출 (file_name, source, file_path 순서로 시도)
                    source_filename = Path(
                        doc.metadata.get("file_name") or
                        doc.metadata.get("source") or
                        doc.metadata.get("file_path", f"doc_{uuid.uuid4().hex[:8]}.md")
                    ).name
                    self._save_outputs(tree, doc_nodes, parser, task.collection_name, task.category, source_filename)

                if not self.test_mode:
                    self.index_nodes(all_task_nodes, task.collection_name)

            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()

        print("\n🎉 모든 작업이 완료되었습니다.")

# --- 4. 실행부 ---

if __name__ == "__main__":
    # 설정 초기화
    config = LegalParserConfig(
        TOKENIZER_FILE_PATH="./tokenizer.json", # 실제 파일 경로로 수정 필요
        LEAF_TOKEN_SIZE=512,
        LEAF_TOKEN_OVERLAP=50,
        PARENT_TARGET_TOKEN_SIZE=1500,
        PARENT_MAX_TOKEN_SIZE=2048,
        OUTPUT_DIR="/home/coder/project/data_job/output"
    )
    
    tasks = [
        DocumentTask(
            source="/home/coder/project/data_job/data/laws_storage",
            category="법령",
            collection_name="rag_legal_docs_v1",
            file_type="고시",
            recursive=True
        ),
    ]

    # 프로세서 실행
    processor = LegalDocumentProcessor(config=config, test_mode=True)
    processor.run(tasks)
