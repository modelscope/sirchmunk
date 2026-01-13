import os

# Limit for concurrent RGA requests
GREP_CONCURRENT_LIMIT = int(os.environ.get("GREP_CONCURRENT_LIMIT", "5"))
