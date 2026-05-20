Copyright © 2024 Leibniz Institute of Plant Genetics and Crop Plant Research (IPK), Gatersleben, Germany

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## Installation
The [prediction models](https://github.com/NAMlab/streamlit_deepcre_app/tree/master/models) are stored via [git lfs](https://git-lfs.com/) so make sure you have git lfs installed and set up before cloning the repository, otherwise you will only get pointer text files.

Then, to install required packages after cloning, run 

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
