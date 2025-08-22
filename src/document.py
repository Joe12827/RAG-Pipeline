class Document:
    def __init__(self, vector: list, text: str = "", source: str = ""):
        self.vector = vector
        self.text = text
        self.source = source

    def get_metadata(self):
        return {
            "text": self.text,
            "source": self.source
        }

    def __repr__(self):
        return f"Document(text={self.text[:50]!r}..., vector={self.vector!r}, metadata={self.metadata!r})"