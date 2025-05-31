FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -a

COPY app.py features.json modelCard.json .
COPY production_models/* .
COPY ml_pipeline ./ml_pipeline

EXPOSE 5000

CMD ["conda", "run", "--no-capture-output", "-n", "mlops-project", "python", "app.py"]







# FROM continuumio/miniconda3

# WORKDIR /app

# COPY environment.yml .

# RUN conda env create -f environment.yml

# SHELL ["conda", "run", "-n", "mlops-project", "/bin/bash", "-c"]

# COPY app.py .

# COPY production_models/* .

# COPY features.json .

# EXPOSE 5000

# CMD ["conda", "run", "-n", "mlops-project", "python", "app.py"]


