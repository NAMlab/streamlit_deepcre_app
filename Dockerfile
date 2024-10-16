FROM python:3.9-buster

# Install micromamba for dependency management
SHELL ["/bin/bash", "-c"]
RUN wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
RUN micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
RUN . ~/.bashrc

# Clone and set up the app
WORKDIR /app
RUN git clone https://github.com/PelFritz/streamlit_deepcre_app .
RUN micromamba env create -f conda-env.yml -y

# Activate the environment and run the app on startup
CMD [ "micromamba", "run", "-n", "deepCRE_CPU", "streamlit", "run", "app.py"]
