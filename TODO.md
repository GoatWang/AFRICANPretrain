# TODO
1. Try full dataset using A100 GPU
2. Try testing it on BlueCrystol
3. check output scale and logit scale of the original clip model (line 88)

# Progress
1. tested on batch_size=1 training_test_size=32 example, got all zeros prediction. (See [Colab Link](https://colab.research.google.com/drive/1CloDFzKibHliU_A1s3J5m-d1d6cfFfaq?usp=drive_link))


# Try
1. Loss infoNCE: 
    - https://github.com/RElbers/info-nce-pytorch
    - https://kevinmusgrave.github.io/pytorch-metric-learning/losses/
2. augumentation: 
    - Image: https://arxiv.org/pdf/2002.05709.pdf
        > 2. Method: A stochastic data augmentation module that transforms
    any given data example randomly resulting in two correlated views of the same example, denoted x˜i and x˜j ,
    which we consider as a positive pair. In this work, we
    sequentially apply three simple augmentations: random
    cropping followed by resize back to the original size, random color distortions, and random Gaussian blur. As
    shown in Section 3, the combination of random crop and
    color distortion is crucial to achieve a good performance.
    - Video: https://arxiv.org/pdf/2008.03800.pdf