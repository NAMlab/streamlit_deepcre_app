## Installation
To install required packages, run 

```
conda create env -f conda-env.yml
```

## Usage
To start the app, run

```
streamlit run app.py
```

## Container
You can get the app as a Docker container from https://hub.docker.com/r/thyra/deepcre_cpu . Make sure you bind the port 8501 when you run it to make the streamlit app accessible from the outside.
