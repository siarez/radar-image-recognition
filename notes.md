Data augmentation:
* H, V, 90, 180, 270, T, T',
* scaling
* noise?(wih dropout that might not be necessary)
* Feature Standardization
* whitening
* more rotation
* shift translate - 15px doable
feed inc_angle
Try the reconstruction idea from Capsules paper
Think and try mean and std before and after logistic
Derive a formula for sensibly adjusting guesses according to confidence
Batchnorm layer settings? Remove to see if it helps

weird images:
3907
2691
8316
They seem to be translated/rotated
Two peaks in histogram
1666 looks very much like an iceberg

dout: 0.1 [Epoch: 247 loss: 0.2920 | acc: 0.8704 | vloss: 0.3403 | vacc: 0.8625 ]
dout: 0.2 [Epoch: 249 loss: 0.3447 | acc: 0.8465 | vloss: 0.3259 | vacc: 0.8656 ]