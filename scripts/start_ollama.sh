# TODO: 需要解决任意路径启动的问题

# TODO:  启动时需要支持用户可选，ollama或者自定义的base url(openai client方式)

# Run with high-performance settings
# Note: 在ollama_server.py同级目录下运行此脚本
uvicorn ollama_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 8 \
  --loop uvloop \
  --http httptools \
  --backlog 2048 \
  --timeout-keep-alive 60 \
  --forwarded-allow-ips '*'
  --app-dir ./
