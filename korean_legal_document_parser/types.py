from dataclasses import dataclass


@dataclass
class DocumentTask:
    """Input source and target collection for one legal document processing job."""

    source: str
    category: str
    collection_name: str
    file_type: str = "문서"
    recursive: bool = True


@dataclass
class LegalParserConfig:
    """Runtime configuration for Korean legal document parsing."""

    TOKENIZER_FILE_PATH: str = "./tokenizer.json"
    LEAF_TOKEN_SIZE: int = 512
    LEAF_TOKEN_OVERLAP: int = 50
    PARENT_TARGET_TOKEN_SIZE: int = 1500
    PARENT_MAX_TOKEN_SIZE: int = 2048
    OUTPUT_DIR: str = "./output"
