## Installation
To install required packages, run 

```
conda env create -f conda-env.yml
```

## Usage
To start the app, run

```
streamlit run app.py
```

## Container
You can get the app as a Docker container from https://hub.docker.com/r/thyra/deepcre_cpu . Make sure you bind the port 8501 when you run it to make the streamlit app accessible from the outside.

## Providing selectable Genomes
If you want to provide genomes for the users to select from without having to upload them themselves, you can add them to the `genomes` folder.
Please refer to the README in that folder for more details.
If you are deploying in a container as we do, you can mount (and override) that directory so you don't have to include these large files in the container itself.