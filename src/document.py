class Document:
    def __init__(self, text: str, vector: list, metadata: dict):
        self.text = text
        self.vector = vector
        self.metadata = metadata

    def __repr__(self):
        return f"Document(text={self.text[:50]!r}..., vector={self.vector!r}, metadata={self.metadata!r})"