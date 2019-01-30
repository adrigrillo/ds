function [y_fft] = plot_fft(y,fs,axis_lim)
%CALCULATE_PLOT_FFT Calculates and plot the fourier transform of a given 
%signal
%   y: the signal to calculate the transform and plot
%   fs: the sample frequency of the signal
%   axis_lim: list of the axis limits, false if not limit are desired
    signal_length=length(y);% Length of signal
    y_fft=fft(y);
    P2 = abs(y_fft/signal_length);
    P1 = P2(1:signal_length/2+1);
    P1(2:end-1)=2*P1(2:end-1);
    f=fs*(0:(signal_length/2))/signal_length;
    figure
    plot(f,P1);
    title('Single-Sided Amplitude Spectrum of X(t)');
    xlabel('f (Hz)');
    ylabel('|P1(f)|');
    if sum(axis_lim)~=0
        axis(axis_lim)
    end
end

