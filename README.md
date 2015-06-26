# thesis
Repo for Master's Thesis at The Cooper Union

## Basic ideas
- more observations (more recordings)
- implies more information
- implies higher quality

The thesis is going to concern getting from more information to higher quality. For speech, this would result in better word error rates, less distortion, higher perceived quality. For other audio like music, this would result in less distortion/noise and higher perceived quality.

## Getting there

First steps:
- read some Hinton papers (the guy who did lots of autoencoder stuff and denoising)
- setup Theano
- take known signals (sine waves, sample audio clips)
- apply random channel
- add gaussian noise
- split into windowed frames
- pass each through autoencoder
- (sequence of autoencoders)