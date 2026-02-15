import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.services.rag_service import answer_query, rag_service

router = APIRouter()


# ==========================================================
# 1️⃣ Normal (Non-Streaming) Endpoint
# ==========================================================
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return answer_query(request.query)


# ==========================================================
# 2️⃣ Streaming Endpoint (SSE Format)
# ==========================================================
@router.post("/chat/stream")
def chat_stream(request: ChatRequest):

    def event_generator():
        try:
            for chunk in rag_service.stream_answer(request.query):
                # Send as JSON chunk (frontend friendly)
                yield f"data: {json.dumps({'token': chunk})}\n\n"

            # Send completion event
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
