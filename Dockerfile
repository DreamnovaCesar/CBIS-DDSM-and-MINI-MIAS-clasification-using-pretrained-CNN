FROM python:3.14.2

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./Final_Code_Main.py"]
