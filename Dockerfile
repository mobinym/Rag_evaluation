FROM python:3.12-slim


ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


WORKDIR /app

ENV PIP_INDEX_URL=http://nexus.aiopt.io:8081/repository/repo-pypi/simple/
ENV PIP_TRUSTED_HOST=nexus.aiopt.io:8081
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


EXPOSE 8000


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]