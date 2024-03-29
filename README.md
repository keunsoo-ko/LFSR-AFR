![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)



# LFSR-AFR

Keunsoo Ko, Yeong Jun Koh, SoonKeun Chang, and Chang-Su Kim

Official PyTorch Code for 
"Light Field Super-Resolution via Adaptive Feature Remixing, IEEE Trans. Image Process., vol. 30, pp. 4114-4128, Apr. 2021"

### Requirements
- PyTorch 1.3.1 (Other versions can cause different results)
- python 3.7

### Installation
Download repository:
```
    $ git clone https://github.com/keunsoo-ko/LFSR-AFR.git
```
Download [pre-trained model](https://drive.google.com/file/d/15Y5KrMbD1lpMN2jUeV9KLChaERG3q_Zf/view?usp=sharing) parameters

### Usage
Run Test for the spatial super resolution on the HCI dataset with the factor x2:
```
    $ python demo.py --mode SR --path LFSR-AFR.pth(put downloaded model path)
```
Run Test for the angular super resolution on the HCI dataset with the factor x2:
```
    $ python demo.py --mode AR --path LFSR-AFR.pth
```
