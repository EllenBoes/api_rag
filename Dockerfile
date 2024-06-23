FROM python:3.9-slim

WORKDIR /app

COPY . /app

ENV CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1

RUN pip uninstall llama-cpp-python

RUN pip install -r requirements.txt

CMD ["uvicorn", "api_rag:app", "--reload", "--port", "8080", "--host","0.0.0.0"]