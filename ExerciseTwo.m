function ExerciseTwo

%Read the input image
inputImage = double(imread("ampelmaennchen.png"));

%Convert the input image to grey scale image and representing with double
%precision float
greyImage = rgb2gray(inputImage);

figure; imshow(greyImage);
title("Grayscale Image");

%Assuming a standard deviation of 0.5 in x and y direction for computing
%kernel

sigmaX = 0.5;
sigmaY = 0.5;

kernelRadius = 2; % Asumming a size of 5x5 for both x and y drections

%Task A: Gradient of Gaussian Filtering
%-------------------------------------------------------------------------

%a. Compute the GOG kernels for convolution in x and y direction
[Gx, Gy] = ComputeGOGFilterKernels(sigmaX, sigmaY, kernelRadius);

%b. Compute the Image Gradient in two directions
[Ix, Iy] = ComputeImageGradient(greyImage, Gx, Gy);

%c. Compute and visualize the gradient magnitude
G = sqrt(Ix.^2 + Iy.^2);
figure;
imshow(G);
title('Gradient Magnitude Image');

%Task B: Forstner Interest Operator
%-----------------------------------------------------------------------

%a. Compute the autocorrelation matrix M
M = ComputeAutocorrelationMatrix(Ix, Iy);

%b. Compute Cornerness and Roundness and Plot the Values
[W, Q] = ComputeCornernessandRoundness(M);

figure;
subplot(1,2,1);
imshow(W);
title('Cornerness');
colormap(jet);
colorbar;
set(gca, 'Position', [0.01, 0.1, 0.4, 0.8]); % Adjust the position and size of the subplot axes

subplot(1,2,2);
imshow(Q);
title('Roundness');
colormap(jet);
colorbar;
set(gca, 'Position', [0.5, 0.1, 0.4, 0.8]); % Adjust the position and size of the subplot axes

%c. Derive a binary mask of potential interest points by simultenously applying thresholds
[Wmask, Qmask] = DeriveBinaryMask(W, Q);

%d. Plot the Overlay of the initial input image with the detected points
PlotOverlay(inputImage, Wmask);


end

%Implementing a function to compute GOG Filter Kernels for Convolution
function [Gx, Gy] = ComputeGOGFilterKernels(sigmaX, sigmaY, kernelRadius)

%Defining a matrix Cx with (2r+1) row and column and Cy
Cx = meshgrid(-kernelRadius:kernelRadius, zeros(1, 2*kernelRadius + 1));
Cy = Cx';

%Computing the GOG filter kernels for x and y direction
Gx = (-Cx./(2 * pi * sigmaX^4)) .* exp(-(Cx.^2 + Cy.^2)/(2 * sigmaX^2));
Gy = (-Cy./(2 * pi * sigmaY^4)) .* exp(-(Cx.^2 + Cy.^2)/(2 * sigmaY^2));

disp("The GOG Filter Kernel in the x-direction is Gx:");
disp(Gx);

disp("The GOG Filter Kernel in the y-direction is Gy:");
disp(Gy);

end

%Function Implemented to determine the Image Gradients
function [Ix, Iy] = ComputeImageGradient(greyImage, Gx, Gy)

%perform convolution with Gx and Gy filter
Ix = conv2(greyImage, Gx, "valid");
Iy = conv2(greyImage, Gy, "valid");


end

%Implementing a function to compute the autocorrelation matrix
function M = ComputeAutocorrelationMatrix(Ix, Iy)

% Defining the weighting function
w = ones(5);

% Initializing the matrix M
[rows, cols] = size(Ix);
M = zeros(rows, cols, 2, 2);

% Computing the Autocorrelation Matrix for each Pixel
for i = 1:rows
    for j = 1:cols

        % Extracting the window around the current pixel
        windowIx = Ix(max(1, i - 2):min(rows, i + 2), max(1, j - 2):min(cols, j + 2));
        windowIy = Iy(max(1, i - 2):min(rows, i + 2), max(1, j - 2):min(cols, j + 2));

        % Extracting the corresponding window from the weighting function
        % Adjusting the size of the weighting function to match the window
        w_window = w(1:size(windowIx,1), 1:size(windowIx,2));

        % Computing the elements of the autocorrelation matrix M
        M(i, j, 1, 1) = sum(sum(w_window .* windowIx .* windowIx));
        M(i, j, 1, 2) = sum(sum(w_window .* windowIx .* windowIy));
        M(i, j, 2, 1) = sum(sum(w_window .* windowIy .* windowIx));
        M(i, j, 2, 2) = sum(sum(w_window .* windowIy .* windowIy));
    end
end

end

%Implementing a function to compute Cornerness and Roundness
function [W, Q] = ComputeCornernessandRoundness(M)

%Extracting the rows and columns from the Autocorrelation Matrix
[row, col, ~, ~] = size(M);

%Initializing the variables W and Q for storing cornerness and randomness
W = zeros(row, col);
Q = zeros(row, col);

for i = 1:row
    for j = 1: col

        %Extracting the Autocorrelation Matrix for the current pixel since
        % The squeeze function is used to convert the 4D matrix to 2D
        A = squeeze(M(i, j, :, :));

        %We compute the determinant and trace of the matrix
        det_M = det(A);
        trace_M = trace(A);

        %We then proceed to compute the cornerness and roundness
        W(i, j) = det_M/trace_M;
        Q(i, j) = (4* det_M)/(trace_M^2);
    end
end

end

%implementing a function to derive binary mask and point of interest
function [Wmask, Qmask] = DeriveBinaryMask(W, Q)

tw = 0.004; % As per definition in task sheet
tq = 0.5;

%Derive binary mask
Wmask = W > tw;
Qmask = Q > tq;

end

%Implementing a function to plot overlay
function PlotOverlay(inputImage, Wmask)

% Plot an overlay of the initial input image with the detected points
figure;
imshow(inputImage/255); % Convert to double in the range [0,1]
hold on;
[rows, cols] = size(Wmask);
[x, y] = find(Wmask); % Find the indices of the nonzero elements
scatter(y, x, 'r', 'filled'); % Plot the detected points
hold off;
title('Detected Points Overlay');

end



