## Inpainting Spatiotemporal Urban Data with 3D Partial Convolutions

We adapt partial convolution architectures proposed for image inpainting on the web to operate on 3D histograms (2D space + 1D time) commonly used for data exchange in urban settings.  Urban data is inconsistently released across cities, variables, and neighborhoods, leading to disproportionate research attention on the relatively few complete, high-quality datasets available. To improve coverage and utility, we adapt image neural in-painting techniques to automatically interpolate missing regions of spatiotemporal data. To address the challenge of skewness in urban data, we:
- use 3D partial convolutions to improve detection of transient events in the sparse regions
- propose biased masking (as opposed to random masking) to encourage the model to attend to dense regions.


## Inpainting Results:
- Top row: hourly NYC taxi trip counts
- Bottom row: hourly NYC bikeshare counts

![alt text](https://github.com/BeanHam/urban-inpainting/blob/main/imgs/1-teaser.png)

## Masking Techniques
- Random Masking: randomly select a starting point from the image -> random-walk algorithm. Since the data distribution is highly imbalanced, it is likely that the mask will not cover any dense region.
![alt text](https://github.com/BeanHam/urban-inpainting/blob/main/imgs/5-random-masking.png)

- Biased Masking: detect the dense regions in the image -> randomly select a starting point from one of those dense regions -> random-walk algorithm. Biased masking will ensure the coverage of dense region, while also covering the sparse region with random-walk.
![alt text](https://github.com/BeanHam/urban-inpainting/blob/main/imgs/6-biased-masking.png)
