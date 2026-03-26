# Imagemage #
Imagemage contains experimental and exploratory image transformations, all callable with the `imagemage.py` API.

## Transformations ##
The following transformations are supported:
- ECB Encryption
- Independent Multi-Channel Low-Rank Approximation
- Mono

The following transformations are planned:
- Quaternion SVD Low-Rank Approximation
- Fuzzy Autoencoder
- Random Channel Walk
- Conditioned Diffusion

## Installation ## 
Create a fresh virtual environment of your choice to install Imagemage dependencies.

For ease of use, I recommend using a simple virtual environment:
```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```


## Usage ## 
Before usage, make sure you're using the correct virtual environment. For example,
```
. .venv/bin/activate
```

You may add an image to the library, thereby allowing it to be transformed in-library by Imagemage, by running:
```
python imagemage.py --add PATH/TO/example.png
```
You may then transform the image by adding additional flags to `imagemage.py`. Each additional flag corresponds to a transformation, and multiple flags can be run at once to run each specified transformation in sequence. For example,
```
python imagemage.py --input example --svd True --ecb True
```

The following flags are available:
```
--svd : Independent Multi-Channel Low-Rank Approximation
--ecb : ECB (Electronic Code Book) Encryption
--mono : RGB Image to Mono
```