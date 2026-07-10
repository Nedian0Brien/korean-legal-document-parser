import json
import os
from typing import Any, List

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
import qdrant_client

from .types import DocumentTask, LegalParserConfig


class LegalDocumentStorage:
    """Local report output and Qdrant indexing for parsed legal documents."""

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

    def setup_qdrant_collections(self, tasks: List[DocumentTask]) -> None:
        if self.test_mode or not self.qdrant_client:
            return

        unique_collections = {task.collection_name for task in tasks}
        try:
            vector_size = len(Settings.embed_model.get_text_embedding("test"))
        except Exception:
            vector_size = 768

        print("🔄 컬렉션 초기화 작업을 수행합니다...")
        for col_name in unique_collections:
            if self.qdrant_client.collection_exists(col_name):
                self.qdrant_client.delete_collection(col_name)
                print(f"   🗑️  기존 컬렉션 삭제됨: {col_name}")

            self.qdrant_client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"   ✅ 새 컬렉션 생성됨: {col_name}")

    def index_nodes(self, nodes: List[BaseNode], collection_name: str) -> None:
        if self.test_mode:
            return

        print(f"🚀 '{collection_name}' 컬렉션에 {len(nodes)}개 노드 업로드 시작...")
        try:
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                enable_hybrid=False,
                batch_size=64,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
            print("✅ 업로드 완료.")
        except Exception as e:
            print(f"❌ 업로드 실패: {e}")

    def save_outputs(
        self,
        tree: Any,
        doc_nodes: List[TextNode],
        parser: Any,
        collection_name: str,
        category: str,
        source_filename: str,
    ) -> None:
        target_dir = os.path.join(self.config.OUTPUT_DIR, collection_name, category)
        os.makedirs(target_dir, exist_ok=True)
        base_name = os.path.splitext(source_filename)[0]

        md_content = parser.tree_to_structured_md(tree, doc_nodes)
        with open(os.path.join(target_dir, f"{base_name}_structured.md"), "w", encoding="utf-8") as f:
            f.write(md_content)

        json_path = os.path.join(target_dir, f"{base_name}_chunks.json")
        chunks_data = []
        for node in doc_nodes:
            rels = {}
            for key, value in node.relationships.items():
                if value is None:
                    rels[str(key.value)] = None
                elif isinstance(value, list):
                    rels[str(key.value)] = [item.node_id if item else None for item in value]
                else:
                    rels[str(key.value)] = value.node_id

            chunks_data.append({
                "id": node.node_id,
                "text": node.text,
                "metadata": node.metadata,
                "relationships": rels,
            })

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=4)
