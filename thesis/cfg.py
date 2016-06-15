minibatches = 1
examples_per_minibatch = 16
framelength = 256
overlap = framelength/2
freq_bins = 256
time_bins = 64

percent_background_latents = 0.25
percent_noise_only_examples = 0.25

lambduh = 0.75

fs = 44100

niter_pretrain = 500
niter_finetune = 500
