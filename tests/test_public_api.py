def test_public_api_exports_parser_types():
    from korean_legal_document_parser import DocumentTask, LegalParserConfig

    assert DocumentTask.__name__ == "DocumentTask"
    assert LegalParserConfig.__name__ == "LegalParserConfig"
