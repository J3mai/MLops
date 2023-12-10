FROM python:3.10.13

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

#EXPOSE 8081

CMD ["streamlit", "run", "app.py"]

#CMD streamlit run --server.port $PORT app.py
