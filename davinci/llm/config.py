# LM Studio connection defaults — edit here or override via env vars
import os

LMS_HOST = os.environ.get("LMS_HOST", "localhost")
LMS_PORT = int(os.environ.get("LMS_PORT", "1234"))

# Model size routing hints (substrings matched case-insensitively against model name)
LARGE_MODEL_HINTS = ["70b", "72b", "80b", "65b"]
SMALL_MODEL_HINTS = ["7b", "8b", "9b", "13b", "14b"]
