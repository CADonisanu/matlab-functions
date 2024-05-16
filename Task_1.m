% Reading the audio signal from "task1sig.wav" file
[audioSignal, Fs] = audioread('task1sig.wav');

% Plotting the audio signal in the time domain
t = (0:length(audioSignal)-1)/Fs;
figure;
plot(t, audioSignal);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Audio Signal in Time Domain');
grid on;

% Displaying the top 10 maximum magnitude values
[sortedValues, sortedIndex] = sort(abs(audioSignal), 'descend');
top10 = sortedValues(1:10);
nativeValues = audioSignal(sortedIndex(1:10));
floatValues = double(nativeValues);
normalisedValues = nativeValues / max(abs(audioSignal));
disp('Top 10 Maximum Magnitude Values:');
disp('Native Values    Float Values    Normalised Values');
for i = 1:10
    fprintf('%10.6f    %10.6f    %10.6f\n', nativeValues(i), floatValues(i), normalisedValues(i));
end

% Creating an FFT based spectrum diagram of the signal
N = length(audioSignal);
Y = fft(audioSignal);
f = (0:(N-1))*(Fs/N);
P = abs(Y)/N;
figure;
plot(f(1:round(N/2)+1), P(1:round(N/2)+1)); % Plot only the first half of the FFT result
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('FFT of Audio Signal');
xlim([0, Fs/2]);
grid on;

% Applying filters to remove background noise
filteredSignal = applyFilters(audioSignal, Fs);

% Saving the filtered audio
audiowrite('filteredAudio.wav', filteredSignal, Fs);

% Playing the audio to demonstrate the noise has been removed
disp('Playing original audio...');
sound(audioSignal, Fs);
pause(length(audioSignal)/Fs + 2);
disp('Playing filtered audio...');
sound(filteredSignal, Fs);
pause(length(filteredSignal)/Fs + 2);

% FFT to show the noise has been successfully removed
Y_filtered = fft(filteredSignal);
P_filtered = abs(Y_filtered)/N;
figure;
plot(f(1:round(N/2)+1), P_filtered(1:round(N/2)+1)); % Plot only the first half of the FFT result
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('FFT of Filtered Audio Signal');
xlim([0, Fs/2]);
grid on;

% Spectrogram Analysis
figure;
spectrogram(audioSignal, 256, 250, 256, Fs, 'yaxis');
title('Spectrogram of Original Audio Signal');
colorbar;

% Noise Reduction using Spectral Subtraction
denoisedSignal = spectralSubtraction(audioSignal, Fs);
denoisedSignal = real(denoisedSignal); % Ensure signal is real and floating point
% Saving the denoised audio
audiowrite('denoisedAudio.wav', denoisedSignal, Fs);
disp('Playing denoised audio...');
sound(denoisedSignal, Fs);
pause(length(denoisedSignal)/Fs + 2);

% Dynamic Range Compression
threshold = 0.1;
ratio = 4;
compressedSignal = dynamicRangeCompression(denoisedSignal, Fs, threshold, ratio);
compressedSignal = real(compressedSignal); % Ensure signal is real and floating point
% Save the compressed audio
audiowrite('compressedAudio.wav', compressedSignal, Fs);
disp('Playing compressed audio...');
sound(compressedSignal, Fs);
pause(length(compressedSignal)/Fs + 2);

% Echo Cancellation
delay = 0.5;  % 500 ms delay
alpha = 0.5;
echoCancelledSignal = echoCancellation(compressedSignal, Fs, delay, alpha);
echoCancelledSignal = real(echoCancelledSignal); % Ensure signal is real and floating point
% Saving the echo-cancelled audio
audiowrite('echoCancelledAudio.wav', echoCancelledSignal, Fs);
disp('Playing echo-cancelled audio...');
sound(echoCancelledSignal, Fs);
pause(length(echoCancelledSignal)/Fs + 2);

% Function for Spectral Subtraction
function denoisedSignal = spectralSubtraction(noisySignal, Fs)
    frameSize = 256;
    overlap = frameSize / 2;
    [S, ~, ~] = stft(noisySignal, Fs, 'Window', hamming(frameSize, 'periodic'), 'OverlapLength', overlap, 'FFTLength', frameSize);
    S = S ./ max(abs(S(:)));
    noiseEstimate = mean(abs(S(:, 1:10)), 2);
    S_denoised = abs(S) - noiseEstimate;
    S_denoised = max(S_denoised, 0) .* exp(1j * angle(S));
    denoisedSignal = istft(S_denoised, Fs, 'Window', hamming(frameSize, 'periodic'), 'OverlapLength', overlap, 'FFTLength', frameSize);
end

% Function for Dynamic Range Compression
function compressedSignal = dynamicRangeCompression(signal, ~, threshold, ratio)
    gain = 1 / ratio;
    compressedSignal = signal;
    compressedSignal(abs(signal) > threshold) = threshold + gain * (compressedSignal(abs(signal) > threshold) - threshold);
end

% Function for Echo Cancellation
function echoCancelledSignal = echoCancellation(signal, Fs, delay, alpha)
    delaySamples = round(delay * Fs);
    h = [1; zeros(delaySamples - 1, 1); -alpha];
    echoCancelledSignal = filter(h, 1, signal);
end

% Function for applying notch and low-pass filters
function filteredSignal = applyFilters(audioSignal, Fs)
    % Design Notch Filter to remove 60 Hz power line noise
    wo = 60 / (Fs / 2);  % Normalized frequency
    bw = wo / 35;        % Bandwidth
    [b_notch, a_notch] = iirnotch(wo, bw);

    % Display the notch filter transfer function
    fvtool(b_notch, a_notch);
    title('Notch Filter Transfer Function');

    % Design Low-Pass Filter to remove high-frequency noise
    Fc = 1000;  % Cutoff frequency in Hz
    [b_low, a_low] = butter(6, Fc / (Fs / 2));

    % Display the low-pass filter transfer function
    fvtool(b_low, a_low);
    title('Low-Pass Filter Transfer Function');

    % Apply the notch filter to the audio signal
    filteredSignal = filter(b_notch, a_notch, audioSignal);

    % Apply the low-pass filter to the already notch-filtered signal
    filteredSignal = filter(b_low, a_low, filteredSignal);

    % Saving the filtered audio for comparison
    audiowrite('filteredAudio.wav', filteredSignal, Fs);
end
