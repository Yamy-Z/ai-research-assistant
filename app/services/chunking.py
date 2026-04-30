from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # If not the last chunk, try to break at sentence boundary
        if end < text_length:
            # Look for sentence endings in the overlap region
            sentence_ends = ['. ', '! ', '? ', '\n']
            best_end = end
            
            for i in range(end, max(start + chunk_size - overlap, start), -1):
                if any(text[i:i+2].startswith(se) for se in sentence_ends):
                    best_end = i + 1
                    break
            
            end = best_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end < text_length else text_length
    
    return chunks