# Differentiable KS Guitar Synthesizer

## Result Spectrogram and Audio Preview
https://bottle-water.github.io/DDSPAcousticGuitar/

## Requirements

- RunPod pod with PyTorch 2.7.x template
- Network Volume mounted at `/workspace`

## Setup

```bash
cd /workspace

# clone the dependency
git clone https://github.com/ptablasdpaula/diffKS_torchLPC
cd diffKS_torchLPC && git submodule update --init --recursive && cd ..

# clone this repo
git clone https://github.com/Bottle-Water/AcousticGuitarDDSP.git
# add lib to python path — do this before anything else
echo "/workspace/diffKS_torchLPC" >> $(python -c "import site; print(site.getsitepackages()[0])")/diffKS_path.pth

move `data` directory to root
move `guitar_poc_final.pth` to project root if you want to synthesize without training

# install deps
# WARNING: do NOT run the lib's requirements.txt directly — it will replace torch
pip install torchlpc==0.6 soundfile scipy librosa
pip install git+https://github.com/patrick-kidger/torchcubicspline.git@d16c6bf5b63d03dbf2977c70e19a320653b5e4a8
```

## Data

Place mono WAV files in `data/`. Filename must embed the fundamental frequency:
```
guitar80Hz.wav
guitar220Hz.wav
guitar657Hz.wav
...
```

## Train

```bash
# Training (~5 min on A4000)
python train.py 


```


## Synthesize

```bash
# Rendering results
python render.py
```

Any f0 can be passed in render.py, the model generalizes
beyond the training notes!
