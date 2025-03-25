FROM nvidia/cuda:12.8.1-base-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    libglib2.0-0 libsm6 libxext6 libxrender1 gcc && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pipenv
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

WORKDIR /code

COPY Pipfile Pipfile.lock ./

RUN pipenv install --deploy --system --skip-lock

COPY /app /code/app

ENV PYTHONPATH=/code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
