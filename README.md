## Adapting to Skew: Imputing Spatiotemporal Urban Data with 3D Partial Convolutions and Biased Masking

We adapt partial convolution architectures proposed for image inpainting on the web to operate on 3D histograms (2D space + 1D time) commonly used for data exchange in urban settings.  Urban data is inconsistently released across cities, variables, and neighborhoods, leading to disproportionate research attention on the relatively few complete, high-quality datasets available. To improve coverage and utility, we adapt image neural in-painting techniques to automatically interpolate missing regions of spatiotemporal data. To address the challenge of skewness in urban data, we:
- use 3D partial convolutions to train simultaneously in space and time
- focus attention on dense regions by biasing the masks used for training to the skew in the data


## Inpainting Results:
A histogram of taxi pickups in Manhattan.  We adapt imagine inpainting techniques to reconstruct missing and corrupted data in urban settings: The improved model (upper left) uses biased masking and temporal context to capture local effects (red circle).  The basic model (lower left) uses ordinary masking and is insensitive to local effects.  Baseline methods that ignore space (lower middle) or time (lower right) are not competitive.  Classical linear methods such as kriging and inverse-distance weighting (not shown) cannot impute large irregular regions in dynamic settings. 
![alt text](https://github.com/BeanHam/urban-inpainting/blob/main/imgs/teaser.png)

## Masking Techniques
- Random Masking: randomly select a starting point from the image -> random-walk algorithm. Since the data distribution is highly imbalanced, it is likely that the mask will not cover any dense region.
![alt text](https://github.com/BeanHam/urban-inpainting/blob/main/imgs/random-masking.png)

- Biased Masking: detect the dense regions in the image -> randomly select a starting point from one of those dense regions -> random-walk algorithm. Biased masking will ensure the coverage of dense region, while also covering the sparse region with random-walk.
![alt text](https://github.com/BeanHam/urban-inpainting/blob/main/imgs/biased-masking.png)
