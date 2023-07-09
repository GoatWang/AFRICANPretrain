# TODO
- augumentation: 
    - Image: https://arxiv.org/pdf/2002.05709.pdf
        > 2. Method: A stochastic data augmentation module that transforms
    any given data example randomly resulting in two correlated views of the same example, denoted x˜i and x˜j ,
    which we consider as a positive pair. In this work, we
    sequentially apply three simple augmentations: random
    cropping followed by resize back to the original size, random color distortions, and random Gaussian blur. As
    shown in Section 3, the combination of random crop and
    color distortion is crucial to achieve a good performance.
    - Video: https://arxiv.org/pdf/2008.03800.pdf
    
# suspend
- Try testing it on BlueCrystol

# DONE
- Try full dataset using A100 GPU
- check output scale and logit scale of the original clip model (line 88)
- Loss infoNCE: 
    - https://github.com/RElbers/info-nce-pytorch
    - https://kevinmusgrave.github.io/pytorch-metric-learning/losses/



