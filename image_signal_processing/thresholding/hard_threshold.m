function [X] = hard_threshold(X,lambda)
%HARD_THRESHOLD perform hard thresholding in a given signal
%   A function to perform hard thresholding on a 
%   given an input vector X with a given threshold lambda
%   X: signal to apply thresholding
%   lambda: threshold value
    ind=abs(X)<=lambda;
    X(ind)=0;
end