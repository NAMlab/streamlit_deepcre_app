FROM python:3.9-bookworm

# Install curl and git-lfs
RUN apt-get update && apt-get install -y curl git \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba for dependency management
SHELL ["/bin/bash", "-c"]
RUN wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
RUN micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
RUN . ~/.bashrc

# Clone and set up the app
WORKDIR /app
RUN git clone https://github.com/NAMlab/streamlit_deepcre_app .
RUN micromamba env create -f conda-env.yml -y

# Activate the environment and run the app on startup
CMD [ "micromamba", "run", "-n", "deepCRE_CPU", "streamlit", "run", "app.py"]
