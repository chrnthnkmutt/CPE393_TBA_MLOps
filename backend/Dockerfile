FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
COPY app.py .
COPY production_models/* .
COPY features.json .

RUN conda env create -f environment.yml && conda clean -a

RUN echo "conda activate mlops-project" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]

EXPOSE 5000

CMD ["bash", "-c", "conda activate mlops-project && python app.py"]







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


