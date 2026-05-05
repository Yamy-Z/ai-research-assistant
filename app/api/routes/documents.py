from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.database import Document, DocumentChunk
from app.models.schemas import DocumentResponse
from app.utils.logger import setup_logger
from app.services.chunking import chunk_text
from app.services.embedding import get_embedding_service, EmbeddingService
from app.services.vector_store import get_vector_store, VectorStore
from app.services.bm25_search import get_bm25_service, BM25SearchService
import uuid

router = APIRouter(prefix="/documents", tags=["documents"])
logger = setup_logger(__name__)


@router.get("/")
async def list_documents(db: Session = Depends(get_db)):
    """List all documents."""
    documents = db.query(Document).all()
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "created_at": doc.created_at
        }
        for doc in documents
    ]


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    bm25_service: BM25SearchService = Depends(get_bm25_service)
):
    """Upload and process a document."""
    
    # Read file content
    content = await file.read()
    text = content.decode('utf-8')
    
    # Create document record
    doc_id = str(uuid.uuid4())
    document = Document(
        id=doc_id,
        filename=file.filename,
        content=text,
        metadata_={"size": len(text)}
    )
    db.add(document)
    
    # Chunk the document
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_service.embed_batch(chunks)
    
    # Prepare payloads for vector store
    payloads = []
    for idx, chunk in enumerate(chunks):
        chunk_record = DocumentChunk(
            document_id=doc_id,
            content=chunk,
            chunk_index=idx,
            metadata_={"length": len(chunk)}
        )
        db.add(chunk_record)
        
        payloads.append({
            "document_id": doc_id,
            "chunk_id": chunk_record.id,
            "content": chunk,
            "chunk_index": idx,
            "filename": file.filename
        })
    
    # Store in vector database
    vector_store.add_vectors(embeddings, payloads)
    
    db.commit()
    bm25_service.reindex()
    
    logger.info(
        f"Document {file.filename} uploaded: "
        f"{len(chunks)} chunks created and indexed"
    )
    
    return DocumentResponse(
        document_id=doc_id,
        filename=file.filename,
        chunks_created=len(chunks),
        status="success"
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    vector_store: VectorStore = Depends(get_vector_store),
    bm25_service: BM25SearchService = Depends(get_bm25_service)
):
    """Delete a document and its chunks."""
    # Delete from vector store
    vector_store.delete_by_document_id(document_id)
    
    # Delete chunks from DB
    db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).delete()
    
    # Delete document from DB
    result = db.query(Document).filter(Document.id == document_id).delete()
    db.commit()
    bm25_service.reindex()
    
    if result == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"status": "deleted", "document_id": document_id}
