# Inpainting Spatiotemporal Urban Data with 3D Partial Convolutions

We adapt partial convolution architectures proposed for image inpainting on the web to operate on 3D histograms (2D space + 1D time) commonly used for data exchange in urban settings.  Urban data is inconsistently released across cities, variables, and neighborhoods, leading to disproportionate research attention on the relatively few complete, high-quality datasets available. To improve coverage and utility, we adapt image neural in-painting techniques to automatically interpolate missing regions of spatiotemporal data. To address the challenge of skewness in urban data, we:
- use 3D partial convolutions to improve detection of transient events in the sparse regions
- propose biased masking (as opposed to random masking) to encourage the model to attend to dense regions.
