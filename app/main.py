import time
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api.chat import router as chat_router

app = FastAPI(title="Enterprise RAG Platform")

# ✅ Track app start time
start_time = time.time()

# ✅ Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# ✅ Custom metrics endpoint (uptime)
@app.get("/custom-metrics")
def custom_metrics():
    uptime = time.time() - start_time
    return {
        "uptime_seconds": round(uptime, 2),
        "status": "running"
    }

# ✅ Include API routes
app.include_router(chat_router)
