FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "mlops-project", "/bin/bash", "-c"]

COPY app.py .
COPY RF_model_prod.pkl .
COPY features.json .

EXPOSE 5000

CMD ["conda", "run", "-n", "mlops-project", "python", "app.py"]
