clear; close all;

filepath = 'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D1.mat';

load(filepath);

fs = 25000;
dt = 1/fs;
N = length(d);
T = N*dt;

frange = (0:N-1)'/T;

% Compute the real-input FFT
fft_result = fft(d);
% frequencies = fftfreq(length(d), 1);  % Frequency values corresponding to the FFT result

figure (1)
plot(frange, log(fft_result));
xlim([0 fs/2]);
% xlim([104 120]);
xlabel('Frequency [Hz]'); ylabel('Amplitude');