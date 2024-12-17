from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import bpm
from app.core.evaluator import Evaluator
# from app.jobs.scheduler import scheduler  # Import the scheduler setup

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend's URL in production, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include BPM route
app.include_router(bpm.router, prefix="/api/v1")