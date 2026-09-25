def split_text(text:str,chunk_size:int=400,overlap: int = 80) -> list[str]:
    """把长文本切成有重叠的小块。"""
    if not text or not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks