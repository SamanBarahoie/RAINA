from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

@router.post("/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    """Upload a new document to the RAG knowledge base."""
    content = await file.read()
    result = "add_document_to_store(file.filename, content)"
    return {"status": "ok", "inserted": result}
