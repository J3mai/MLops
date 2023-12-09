FROM python:3.10.13

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

EXPOSE 8080

#CMD ["streamlit", "run", "main.py"]

CMD streamlit run --server.port $PORT app.py
