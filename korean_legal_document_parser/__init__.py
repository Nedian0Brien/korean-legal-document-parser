from .types import DocumentTask, LegalParserConfig

__all__ = [
    "DocumentTask",
    "LegalDocumentProcessor",
    "LegalParserConfig",
    "RegalDocumentParser",
]


def __getattr__(name: str):
    if name in {"LegalDocumentProcessor", "RegalDocumentParser"}:
        from src.chunking.LegalDocumentProcessor import (
            LegalDocumentProcessor,
            RegalDocumentParser,
        )

        return {
            "LegalDocumentProcessor": LegalDocumentProcessor,
            "RegalDocumentParser": RegalDocumentParser,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
