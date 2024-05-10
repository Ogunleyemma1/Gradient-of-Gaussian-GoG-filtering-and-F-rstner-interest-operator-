# Gradient-of-Gaussian-GoG-filtering-and-F-rstner-interest-operator-Image-Analysis-and-processing

use the provided image (ampelmaennchen.png)

# Task.1 Gradient of Gaussian (GoG) filtering

a. Compute continuous GoG-filter kernels for convolution in x- and y-direction.
b. Apply these filters to your input image I to derive two gradient images: 𝐼𝑥 and 𝐼𝑦 (one in x- and one in y-direction). Write a function for the convolution of the image with the kernel and ignore the boundaries of the image for simplicity, i.e. no padding needed (you may use built-in convolution function cov2).
c. Compute and visualize the gradient magnitude image 𝐺.

# Task2. Förstner interest operator

Use the gradient images to identify Förstner interest points in your input image.
a. Compute the autocorrelation matrix 𝑀 for each pixel using a moving window 𝑤 of 5×5 pixels. Perform convolution based on this window to include the local neighborhood around each pixel (use 𝐼𝑥, 𝐼𝑦 and ignore the boundaries of the images).
b. Compute the cornerness 𝑤 and roundness 𝑞 from 𝑀 for each pixel and store the values in two matrices 𝑊 and 𝑄. Plot the values in 𝑊 and 𝑄 with an appropriate colormap (imshow, colormap(jet)).
c. Derive a binary mask of potential interest points by simultaneously applying the thresholds 𝑡𝑤=0.004 and 𝑡𝑞=0.5 on 𝑊 and 𝑄, respectively.
d. Plot an overlay of the initial input image with the detected points (plot).


