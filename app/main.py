from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import uvicorn
from typing import List, Dict, Any
import json
import re
from app.models import QueryRequest, QueryResponse, ChunkScore
from app.retrieval import HybridRetriever
from app.llm import AnswerGenerator

app = FastAPI(title="Hybrid RAG System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
retriever = HybridRetriever()
answer_generator = AnswerGenerator()

@app.on_event("startup")
async def startup_event():
    """Initialize the retriever on startup"""
    retriever.initialize()

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Main endpoint for querying the Hybrid RAG system
    """
    start_time = time.time()
    
    try:
        # Retrieve relevant chunks using hybrid approach
        retrieved_chunks = retriever.retrieve(
            query=request.query,
            top_k=request.top_k if request.top_k else 10
        )
        
        # Generate answer using LLM
        answer, sources = answer_generator.generate_answer(
            query=request.query,
            contexts=retrieved_chunks
        )

        # ----- If generate_answer returns an error encoded as a string, convert it to HTTPException -----
        # Example of error string you provided:
        # "Error generating answer: Error code: 429 - {'error': {...}}"
        if isinstance(answer, str) and answer.lower().startswith("error generating answer"):
            # Try to extract "Error code: <code> - <json/dict>"
            m = re.search(r"Error code:\s*(\d+)\s*-\s*(\{.*\})", answer)
            if m:
                status_code = int(m.group(1))
                body_text = m.group(2)
                # Body might be single quotes, convert to valid JSON
                try:
                    error_body = json.loads(body_text)
                except json.JSONDecodeError:
                    # convert single quotes to double quotes as a best-effort and try again
                    try:
                        error_body = json.loads(body_text.replace("'", '"'))
                    except Exception:
                        error_body = body_text  # fallback to raw text
                # If the body contains the 'error' object, surface it; otherwise use body
                detail = error_body.get("error") if isinstance(error_body, dict) and "error" in error_body else error_body
                raise HTTPException(status_code=status_code, detail=detail)
            else:
                # Could not parse structured error — return 500 with the raw message
                raise HTTPException(status_code=500, detail={"error": answer})
        # -----------------------------------------------------------------------------------------------

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Prepare scores for response
        scores = []
        for chunk in retrieved_chunks[:5]:  # Top 5 for response
            scores.append(ChunkScore(
                chunk_id=chunk["chunk_id"],
                dense_score=chunk["dense_score"],
                sparse_score=chunk["sparse_score"],
                final_score=chunk["final_score"]
            ))
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            scores=scores,
            latency_ms=round(latency_ms, 2)
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions (so status codes from above pass through unchanged)
        raise

    except Exception as e:
        # Best-effort extraction of status code and JSON body from arbitrary exception strings
        # (useful when lower-level libraries embed the status and JSON into the exception text)
        err_text = str(e)

        # 1) If the exception object has a status_code attribute, use it.
        status_code = getattr(e, "status_code", None)
        if status_code is not None:
            try:
                # try to get a JSON body from e.response if available
                resp = getattr(e, "response", None)
                if resp is not None:
                    # some SDK exceptions provide a .response object that is JSON-decodable
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = str(resp)
                else:
                    detail = err_text
            except Exception:
                detail = err_text

            raise HTTPException(status_code=status_code, detail=detail)

        # 2) Fallback: try to parse "Error code: <int> - <json/dict>" from the exception text
        m = re.search(r"Error code:\s*(\d+)\s*-\s*(\{.*\})", err_text)
        if m:
            try:
                code = int(m.group(1))
                body_text = m.group(2)
                try:
                    parsed = json.loads(body_text)
                except json.JSONDecodeError:
                    parsed = json.loads(body_text.replace("'", '"'))
                detail = parsed.get("error") if isinstance(parsed, dict) and "error" in parsed else parsed
            except Exception:
                code = 500
                detail = err_text
            raise HTTPException(status_code=code, detail=detail)

        # 3) Last fallback: return 500
        raise HTTPException(status_code=500, detail={"error": err_text})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Hybrid RAG System"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)