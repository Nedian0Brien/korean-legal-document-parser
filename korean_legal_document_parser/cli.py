from .types import DocumentTask, LegalParserConfig
from src.chunking.LegalDocumentProcessor import LegalDocumentProcessor


def main() -> None:
    config = LegalParserConfig(
        TOKENIZER_FILE_PATH="./tokenizer.json",
        LEAF_TOKEN_SIZE=512,
        LEAF_TOKEN_OVERLAP=50,
        PARENT_TARGET_TOKEN_SIZE=1500,
        PARENT_MAX_TOKEN_SIZE=2048,
        OUTPUT_DIR="/home/coder/project/data_job/output",
    )

    tasks = [
        DocumentTask(
            source="/home/coder/project/data_job/data/laws_storage",
            category="법령",
            collection_name="rag_legal_docs_v1",
            file_type="고시",
            recursive=True,
        ),
    ]

    processor = LegalDocumentProcessor(config=config, test_mode=True)
    processor.run(tasks)


if __name__ == "__main__":
    main()
