FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "mlops-project", "/bin/bash", "-c"]

COPY . .

RUN conda run -n mlops-project pip install notebook

EXPOSE 8888

CMD ["conda", "run", "-n", "mlops-project", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
