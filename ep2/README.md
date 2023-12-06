# COIN 🌟

This repo contains a Pytorch implementation of [COIN: COmpression with Implicit Neural representations](https://arxiv.org/abs/2103.03123), including code to reproduce all experiments and plots in the paper.


## Requirements

We ran our experiments with `python 3.8.7` using `torch 1.7.0` and `torchvision 0.8.0` but the code is likely to work with earlier versions too. All requirements can be installed with

```pip install -r requirements.txt```

## Usage

### Representation

To represent the image `kodak-dataset/kodim15.png`, run

```python main.py```


### Plots

To recreate plots from the paper, run

```python plots.py```

See the `plots.py` file to customize plots.

## Acknowledgements

Our benchmarks and plots are based on the [CompressAI](https://github.com/InterDigitalInc/CompressAI) library. Our SIREN implementation is based on [lucidrains'](https://github.com/lucidrains/siren-pytorch) implementation.

## License

MIT
