function compare_plot_fft(signal_1,signal_2,fs,axis_lim)
%CALCULATE_PLOT_FFT Calculates and plot the fourier transform of a given 
%signal
%   signal_1: first signal to calculate the transform and plot
%   signal_2: second signal to calculate the transform and plot
%   fs: the sample frequency of the signals
%   axis_lim: list of the axis limits, false if not limit are desired
    signal_length=length(signal_1); % Length of signal
    f=fs*(0:(signal_length/2))/signal_length;
    % signal 1
    signal_1_fft=fft(signal_1);
    P2_1 = abs(signal_1_fft/signal_length);
    P1_1 = P2_1(1:signal_length/2+1);
    P1_1(2:end-1)=2*P1_1(2:end-1);
    % signal 2
    signal_2_fft=fft(signal_2);
    P2_2 = abs(signal_2_fft/signal_length);
    P1_2 = P2_2(1:signal_length/2+1);
    P1_2(2:end-1)=2*P1_2(2:end-1);
    % plot
    figure
    plot(f,P1_1);
    hold on
    plot(f,P1_2);
    legend('Signal 1', 'Signal 2');
    title('Single-Sided Amplitude Spectrum of X(t)');
    xlabel('f (Hz)');
    ylabel('|P1(f)|');
    if sum(axis_lim)~=0
        axis(axis_lim)
    end
end

