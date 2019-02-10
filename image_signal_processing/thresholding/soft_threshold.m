function [th_X] = soft_threshold(X,lambda)
%soft_threshold perform soft thresholding in a given signal
%   The soft thresholding is also called wavelet shrinkage, 
%   as values for both positive and negative coefficients are being
%   "shrinked" towards zero, in contrary to hard thresholding which 
%   either keeps or removes values of coefficients.
%   In case of image de-noising, you are not working strictly on
%   "intensity values", but wavelet coefficients.
%   X: signal to apply thresholding
%   lambda: threshold value
    th_X=abs(X)-lambda;
    th_X(th_X<0)=0;
    th_X=th_X.*sign(X);
end

