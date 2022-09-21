# MIND-neural-decoder

This repository contains the official TensorFlow implementation of the following paper:

MIND: Maximum Mutual Information Based Neural Decoder - https://ieeexplore.ieee.org/document/9895352

If you use the repository for your experiments, please cite the paper.

<img src="https://github.com/tonellolab/MIND-neural-decoder/blob/main/teaser_arch.jpg" width=800>


The paper presents a neural decoding strategy that is based on the mutual information maximization, denoted as MIND.
MIND is a simple feedforward neural network trained to estimate density-ratios and the mutual information. By design, it can provide estimates of the following quantities:
- A-posteriori probabilities
- Bit-error-rate
- Probability of error
- Entropy of the source
- Conditional entropy
- Mutual information

The sample code is developed for the supervised approach (see the paper) and for binary modulation schemes but can easily be extended to M-PAM, M-QAM and more. Coding strategies such as repetition, hamming and convolutional codes are described in the original paper and can be implemented. MIND considers also channel non-linearities.

<img src="https://github.com/tonellolab/MIND-neural-decoder/blob/main/teaser.png" width=800>

Two noise options are available to train your own MIND model:
- AWGN
- Middleton

To train and test a decoder of binary AWGN noisy samples use the following command
> python MIND.py

To train and test a decoder of received samples affected by truncated Middleton noise (a.k.a. Bernoulli-Gaussian) with K=5, use the following command
> python MIND.py --noise Middleton

Training and testing parameters such as training epochs, batch and test sizes can be given as input
> python MIND.py --epochs 500 --batch_size 32 --test_size 10000
