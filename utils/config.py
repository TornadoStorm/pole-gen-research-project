import os

from dotenv import load_dotenv
from segments import SegmentsClient

load_dotenv()

SEGMENTS_CLIENT = SegmentsClient(os.getenv("SEGMENTS_AI_API_KEY"))
