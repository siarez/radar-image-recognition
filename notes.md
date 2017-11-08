Data augmentation:
* H, V, 90, 180, 270, T, T',
* scaling
* Feature Standardization - StdNormalize
* whitening
* more rotation
* shift translate - 15px doable
Remove max and mix inputs in the angle encoding. I think we'll be back to before having the angle in performance.
Concat angle as channel to image and feed to conv net.F
Use RandomChoiceCompose to apply transpose
Use RandomChoiceRotate istead of "range rotate"
feed GMM
Address class imbalance. Although test set may have a different balance

Replace NA with average of ships
Try the reconstruction idea from Capsules paper
Batchnorm layer settings? Remove to see if it helps
Train on whole data before running on test data
save model parameters



weird images:
3907
2691
8316
They seem to be translated/rotated
Two peaks in histogram
1666 looks very much like an iceberg

dout: 0.1 [Epoch: 247 loss: 0.2920 | acc: 0.8704 | vloss: 0.3403 | vacc: 0.8625 ]
dout: 0.2 [Epoch: 249 loss: 0.3447 | acc: 0.8465 | vloss: 0.3259 | vacc: 0.8656 ]
dout: 0.2 - no rotation [Epoch: 249 loss: 0.2550 | acc: 0.8963 | vloss: 0.3291 | vacc: 0.8607 ]
dout: 0.2 - no rotation 16-64 [Epoch: 249 loss: 0.2529 | acc: 0.8941 | vloss: 0.2576 | vacc: 0.9062 ]
dout 0.2, 45deg, first and seconf conv are two, sigmoid after mean and std
[Epoch: 249 loss: 0.3682 | acc: 0.8608 | vloss: 0.3038 | vadjloss: 0.2850 | vacc: 0.8656 | vacc_m: 0.8844 ]
