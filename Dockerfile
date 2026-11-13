FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && ln -s /usr/local/lib/python3.11/site-packages/scikit_learn.libs/libgomp*.so* /usr/lib/libgomp.so.1 \
    && ldconfig

COPY . .

EXPOSE 8008

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]
