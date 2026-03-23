import os

LMS_HOST = os.environ.get("LMS_HOST", "192.168.0.176")
LMS_PORT = int(os.environ.get("LMS_PORT", "1234"))

LARGE_MODEL_HINTS = ["70b", "72b", "80b", "65b"]
SMALL_MODEL_HINTS = ["7b", "8b", "9b", "13b", "14b"]