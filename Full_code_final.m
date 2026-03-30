clc;
clear;
close all;

baseDir = '/Users/ome/Desktop/3-2/EEE 376/Project/Dataset/Final Dataset'; 
subfolders = {'train', 'test','new batch'};
outputFile = 'DSP_Features_Merged(1).xlsx';
all_features = struct();
counter = 1;
fprintf('Starting Full Feature Extraction Pipeline...\n');

for s = 1:length(subfolders)
    folderPath = fullfile(baseDir, subfolders{s});
    files = dir(fullfile(folderPath, '*.wav'));
    
    fprintf('Processing folder: %s (%d files)...\n', subfolders{s}, length(files));
    
    for f = 1:length(files)
        currentFile = fullfile(files(f).folder, files(f).name);
        if contains(files(f).name, 'y', 'IgnoreCase', true)
            label = 'PD';
        else
            label = 'HC';
        end
   
        try
            feat_struct = extract_all_features(currentFile);
            all_features(counter).Filename = files(f).name;
            all_features(counter).Label = label;
            all_features(counter).Set = subfolders{s};
            fn = fieldnames(feat_struct);
            for k = 1:length(fn)
                all_features(counter).(fn{k}) = feat_struct.(fn{k});
            end
            
            fprintf('  [%d/%d] Processed: %s\n', f, length(files), files(f).name);
            counter = counter + 1;
        catch ME
            fprintf('  [ERROR] Failed on %s: %s\n', files(f).name, ME.message);
        end
    end
end
finalTable = struct2table(all_features);
writetable(finalTable, outputFile);
fprintf('\nSuccess! All features saved to: %s\n', outputFile);

function features = extract_all_features(filename)
    % 1. Read & Standardize
    [y, fs] = audioread(filename);
    if size(y, 2) > 1, y = mean(y, 2); end % Mono conversion
    y = y - mean(y);             
    y = y / max(abs(y)); 

    % 2. Framing Parameters
    frame_len = round(0.04 * fs);    
    hop = round(0.01 * fs);        
    num_frames = floor((length(y)-frame_len)/hop) + 1;

    % 3. Storage & VAD Initialization
    is_speech = false(1, num_frames);
    is_voiced = false(1, num_frames);
    
    F0 = nan(1, num_frames);
    HNR_frame = nan(1, num_frames);
    
    % Cycle arrays
    cycle_periods = cell(1, num_frames);
    cycle_amplitudes = cell(1, num_frames);
    
    % Spectral feature arrays
    spec_centroid = nan(1, num_frames);
    spec_spread   = nan(1, num_frames);
    spec_skew     = nan(1, num_frames);
    spec_kurt     = nan(1, num_frames);
    spec_flatness = nan(1, num_frames);
    spec_entropy  = nan(1, num_frames);
    spec_rolloff  = nan(1, num_frames);
    spec_decrease = nan(1, num_frames);
    spec_slope    = nan(1, num_frames);
    spec_flux     = nan(1, num_frames);
    sonorousness  = nan(1, num_frames);
    
    spectra = cell(1, num_frames);

    % VAD Thresholds
    energy_thresh = 0.005; 
    zcr_thresh = 0.2;      

    % 4. Frame-wise Processing
    for i = 1:num_frames
        idx = (i-1)*hop + 1;
        frame = y(idx:idx+frame_len-1);
        frame_win = frame .* hamming(frame_len);

        % --- A. Voice Activity Detection (VAD) ---
        energy = sum(frame.^2) / frame_len;
        zcr = sum(abs(diff(sign(frame)))) / (2 * frame_len);

        if energy > energy_thresh
            is_speech(i) = true; % Active speech detected

            % --- B. Voiced vs Unvoiced Check ---
            [acf, lags] = xcorr(frame_win, 'coeff');
            zlag = find(lags == 0);
            acf_pos = acf(zlag+1:end);
            lags_pos = lags(zlag+1:end);

            minLag = round(fs / 400);
            maxLag = round(fs / 60);
            [pks, locs] = findpeaks(acf_pos(minLag:maxLag));

            if ~isempty(pks)
                [max_peak, id] = max(pks);
                % Threshold: Peak > 0.3 AND ZCR is low
                if max_peak > 0.3 && zcr < zcr_thresh
                    is_voiced(i) = true;
                    actual_loc = locs(id) + minLag - 1;
                    pitch_period = lags_pos(actual_loc);
                    F0(i) = fs / pitch_period;
                    
                    r0 = acf(zlag);         
                    HNR_frame(i) = 10*log10(max_peak / (abs(r0 - max_peak) + eps));

                    % Cycle-level Analysis for Jitter/Shimmer
                    [pks_time, locs_time] = findpeaks(frame,'MinPeakDistance', minLag);
                    if length(locs_time) > 2
                        cycle_periods{i} = diff(locs_time) / fs;
                        cycle_amplitudes{i} = abs(pks_time(1:end-1));
                    end
                end
            end

            % --- C. Spectral Features (On ALL active speech frames) ---
            NFFT = 2^nextpow2(frame_len);
            X = fft(frame_win, NFFT);
            mag = abs(X(1:NFFT/2));
            spectra{i} = mag;
            
            f = (0:length(mag)-1)' * fs / (2*length(mag));
            mag = mag(:); f = f(:);
            mag = mag + eps; % numerical safety
            E = sum(mag);

            spec_centroid(i) = sum(f .* mag) / E;
            spec_spread(i) = sqrt(sum(((f - spec_centroid(i)).^2) .* mag) / E);
            spec_skew(i) = sum(((f - spec_centroid(i)).^3) .* mag) /(E * spec_spread(i)^3);
            spec_kurt(i) = sum(((f - spec_centroid(i)).^4) .* mag) /(E * spec_spread(i)^4);
            spec_flatness(i) = exp(mean(log(mag))) / mean(mag);
            
            p_mag = mag / sum(mag);
            spec_entropy(i) = -sum(p_mag .* log2(p_mag)) / log2(length(p_mag));
            
            cum_energy = cumsum(mag);
            roll_idx = find(cum_energy >= 0.85*E, 1);
            if ~isempty(roll_idx), spec_rolloff(i) = f(roll_idx); end
            
            k_idx = (2:length(mag))';
            spec_decrease(i) = sum((mag(k_idx) - mag(1)) ./ (k_idx-1)) / sum(mag(k_idx));
            
            f2 = f(2:end); mag2 = mag(2:end);
            X_fit = [ones(length(f2),1) f2];
            b = X_fit \ mag2;
            spec_slope(i) = b(2);
            
            if i > 1 && is_speech(i-1) && ~isempty(spectra{i-1})
                prev = spectra{i-1} + eps;
                spec_flux(i) = sqrt(sum((mag - prev(:)).^2));
            end

            f_cut = 1000; 
            low_energy = sum(mag(f <= f_cut));
            sonorousness(i) = low_energy / E;
        end
    end

    % 5. Global Aggregation
    
    % Combine Cycle Data (Only from VOICED frames)
    all_T0 = []; all_amp = [];
    for i = find(is_voiced)
        if ~isempty(cycle_periods{i}) 
            all_T0 = [all_T0; cycle_periods{i}(:)]; 
        end
        if ~isempty(cycle_amplitudes{i})
            all_amp = [all_amp; cycle_amplitudes{i}(:)]; 
        end
    end
    all_T0 = all_T0(:); 
    all_amp = all_amp(:);

    % --- Feature Output Struct Generation ---
    
    % VAD Feature
    total_speech_frames = sum(is_speech);
    if total_speech_frames == 0 
        total_speech_frames = 1; 
    end 
    features.Degree_Unvoiced = (sum(is_speech & ~is_voiced) / total_speech_frames) * 100;

    % Phonation & Noise Features (Mean of VOICED)
    features.Mean_F0 = mean(F0(is_voiced), 'omitnan');
    features.HNR = mean(HNR_frame(is_voiced), 'omitnan');
    features.NHR = 1 / (10^(features.HNR/10));

    % Perturbation Features (Jitter/Shimmer)
    if length(all_T0) > 4 && length(all_amp) > 4
        N = length(all_T0); M = length(all_amp);
        
        % Jitter
        features.Jitter_local = mean(abs(diff(all_T0))) / mean(all_T0) * 100;
        rap_sum = sum(abs(all_T0(2:N-1) - (all_T0(1:N-2) + all_T0(2:N-1) + all_T0(3:N))/3));
        features.RAP = (rap_sum / (N-2)) / mean(all_T0) * 100;
        ppq5_sum = sum(abs(all_T0(3:N-2) - (all_T0(1:N-4) + all_T0(2:N-3) + all_T0(3:N-2) + all_T0(4:N-1) + all_T0(5:N))/5));
        features.PPQ5 = (ppq5_sum / (N-4)) / mean(all_T0) * 100;
        
        % Shimmer
        features.Shimmer_local = mean(abs(diff(all_amp))) / mean(all_amp) * 100;
        apq3_sum = sum(abs(all_amp(2:M-1) - (all_amp(1:M-2) + all_amp(2:M-1) + all_amp(3:M))/3));
        features.APQ3 = (apq3_sum / (M-2)) / mean(all_amp) * 100;
        apq5_sum = sum(abs(all_amp(3:M-2) - (all_amp(1:M-4) + all_amp(2:M-3) + all_amp(3:M-2) + all_amp(4:M-1) + all_amp(5:M))/5));
        features.APQ5 = (apq5_sum / (M-4)) / mean(all_amp) * 100;
        
        % Nonlinear
        features.PVI = std(all_T0) / mean(all_T0);
        T0_log = log(all_T0 / mean(all_T0));
        [counts, ~] = histcounts(T0_log, 20, 'Normalization', 'probability');
        counts(counts == 0) = [];
        features.PPE = -sum(counts .* log(counts));
    else
        features.Jitter_local=0; features.RAP=0; features.PPQ5=0;
        features.Shimmer_local=0; features.APQ3=0; features.APQ5=0;
        features.PVI=0; features.PPE=0;
    end

    % Spectral Features (Mean of ALL SPEECH)
    features.Spec_Centroid = mean(spec_centroid(is_speech), 'omitnan');
    features.Spec_Spread   = mean(spec_spread(is_speech), 'omitnan');
    features.Spec_Skew     = mean(spec_skew(is_speech), 'omitnan');
    features.Spec_Kurt     = mean(spec_kurt(is_speech), 'omitnan');
    features.Spec_Flatness = mean(spec_flatness(is_speech), 'omitnan');
    features.Spec_Entropy  = mean(spec_entropy(is_speech), 'omitnan');
    features.Spec_Rolloff  = mean(spec_rolloff(is_speech), 'omitnan');
    features.Spec_Decrease = mean(spec_decrease(is_speech), 'omitnan');
    features.Spec_Slope    = mean(spec_slope(is_speech), 'omitnan');
    features.Spec_Flux     = mean(spec_flux(is_speech), 'omitnan');
    features.Sonorousness  = mean(sonorousness(is_speech), 'omitnan');
    fn = fieldnames(features);
    for k=1:length(fn)
        if isnan(features.(fn{k})), features.(fn{k}) = 0; end
    end
end


%% Feature Selection (t-SNE)
clc; clear; close all;
filename = '/Users/ome/Downloads/DSP_Features_Merged (1).xlsx';
data = readtable(filename);
labels = categorical(data.Label);
X = table2array(data(:, 5:end));
X_std = zscore(X);
fprintf('Running t-SNE on %d samples...\n', size(X_std, 1));
[Y, loss] = tsne(X_std, 'Algorithm', 'exact', 'Distance', 'euclidean', 'Perplexity', 15);
figure('Name', 't-SNE: HC vs PD Feature Separation', 'Color', 'w');
gscatter(Y(:,1), Y(:,2), labels, 'rb', 'o+');
title('t-SNE Visualization of Speech Features (HC vs PD)');
xlabel('t-SNE dimension 1'); 
ylabel('t-SNE dimension 2');
legend('Healthy Control (HC)', 'Parkinson''s Disease (PD)', 'Location', 'best');
grid on;
fprintf('t-SNE completed with final loss: %.4f\n', loss);

%% Selection part
csv_file = '/Users/ome/Downloads/DSP_Features_Merged (1).xlsx';
fprintf('Loading dataset: %s\n', csv_file);
opts = detectImportOptions(csv_file);
opts.VariableNamingRule = 'preserve';
T = readtable(csv_file, opts);
feat_cols = 5:width(T);
X = table2array(T(:, feat_cols));               
feat_names = T.Properties.VariableNames(feat_cols); 
labels = T.Label;     
[g, class_names] = findgroups(labels); 
y = double(g);
X_norm = zscore(X);
fprintf('Running Feature Selection algorithms...\n');

n_feats = size(X, 2);
score_fisher = zeros(1, n_feats);
mu_all = mean(X_norm);

for i = 1:n_feats
    mu1 = mean(X_norm(y==1, i)); 
    var1 = var(X_norm(y==1, i));
    mu2 = mean(X_norm(y==2, i)); 
    var2 = var(X_norm(y==2, i));
    n1 = sum(y==1); 
    n2 = sum(y==2);
    score_fisher(i) = (n1*(mu1-mu_all(i))^2 + n2*(mu2-mu_all(i))^2) / (n1*var1 + n2*var2);
end
score_ftest = zeros(1, n_feats);
for i = 1:n_feats
    [~, ~, ~, stats] = ttest2(X_norm(y==1, i), X_norm(y==2, i));
    score_ftest(i) = abs(stats.tstat);
end
[idx_relief, weights_relief] = relieff(X_norm, y, 10);
score_relief = zeros(1, n_feats);
score_relief(idx_relief) = weights_relief(idx_relief);
score_fisher = rescale(score_fisher);
score_ftest  = rescale(score_ftest);
score_relief = rescale(score_relief);
figure('Name', 'Feature Selection Scores', 'Color', 'w', 'Position', [50 50 1200 600]);
scores_matrix = [score_fisher; score_ftest; score_relief]';
[~, sort_idx] = sort(mean(scores_matrix, 2), 'descend');
sorted_names = feat_names(sort_idx);
sorted_scores = scores_matrix(sort_idx, :);
b = bar(sorted_scores(1:15, :));
xticklabels(sorted_names(1:15));
xtickangle(45);
ylabel('Normalized Importance Score (0-1)');
title('Top 15 Features by Method');
legend({'Fisher Score', 'F-Test', 'ReliefF'});
grid on;
figure('Name', 'Consensus Heatmap', 'Color', 'w', 'Position', [100 100 1000 800]);
top_k = 20;
heatmap_data = sorted_scores(1:top_k, :);
x_labels = {'Fisher', 'F-Test', 'ReliefF'};
y_labels = sorted_names(1:top_k);
h = heatmap(x_labels, y_labels, heatmap_data);
h.Title = 'Feature Importance Heatmap (Darker = Better)';
h.Colormap = parula;
top_4_indices = sort_idx(1:4);

figure('Name', 'Top 4 Biomarker Separation', 'Color', 'w', 'Position', [150 150 1000 700]);
sgtitle('Distribution of the Top 4 "Consensus" Biomarkers');

for k = 1:4
    subplot(2, 2, k);
    feat_idx = top_4_indices(k);
    data_feat = X(:, feat_idx);
    boxplot(data_feat, labels, 'Colors', 'br');
    set(findobj(gca,'Type','Line'), 'LineWidth', 2);
    title(feat_names{feat_idx}, 'Interpreter', 'none', 'FontSize', 12);
    grid on;
    [~, p] = ttest2(data_feat(y==1), data_feat(y==2));
    xlabel(sprintf('p-value: %.2e', p));
end
function out = rescale(in)
    out = (in - min(in)) / (max(in) - min(in));
end

%% ML
clc; clear; close all;

data = readtable('/Users/ome/Downloads/DSP_Features_Merged (1).xlsx');

X = [data.Spec_Flatness, data.Spec_Spread,data.Spec_Flux, data.Spec_Centroid,data.APQ3,data.Shimmer_local];
y = data.Label;

k = 10;
cv = cvpartition(y, 'KFold', k);

acc = zeros(k, 8);
sens = zeros(k, 8);
spec = zeros(k, 8);

for i = 1:k
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    X_train = X(trainIdx, :); y_train = y(trainIdx);
    X_test = X(testIdx, :);   y_test = y(testIdx);
    
    mdl1 = fitcsvm(X_train, y_train, 'KernelFunction', 'linear', 'Standardize', true);
    pred1 = predict(mdl1, X_test);
    [acc(i,1), sens(i,1), spec(i,1)] = get_metrics(y_test, pred1);
    
    mdl2 = fitcsvm(X_train, y_train, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2, 'Standardize', true);
    pred2 = predict(mdl2, X_test);
    [acc(i,2), sens(i,2), spec(i,2)] = get_metrics(y_test, pred2);
    
    mdl3 = fitcsvm(X_train, y_train, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3, 'Standardize', true);
    pred3 = predict(mdl3, X_test);
    [acc(i,3), sens(i,3), spec(i,3)] = get_metrics(y_test, pred3);
    
    mdl4 = fitcsvm(X_train, y_train, 'KernelFunction', 'gaussian', 'Standardize', true);
    pred4 = predict(mdl4, X_test);
    [acc(i,4), sens(i,4), spec(i,4)] = get_metrics(y_test, pred4);
    
    mdl5 = fitcknn(X_train, y_train, 'NumNeighbors', 1, 'Distance', 'euclidean');
    pred5 = predict(mdl5, X_test);
    [acc(i,5), sens(i,5), spec(i,5)] = get_metrics(y_test, pred5);
    
    mdl6 = fitcknn(X_train, y_train, 'NumNeighbors', 1, 'Distance', 'chebychev');
    pred6 = predict(mdl6, X_test);
    [acc(i,6), sens(i,6), spec(i,6)] = get_metrics(y_test, pred6);
    
    mdl7 = fitcknn(X_train, y_train, 'NumNeighbors', 1, 'Distance', 'minkowski');
    pred7 = predict(mdl7, X_test);
    [acc(i,7), sens(i,7), spec(i,7)] = get_metrics(y_test, pred7);
    
    mdl8 = fitcknn(X_train, y_train, 'NumNeighbors', 1, 'Distance', 'spearman');
    pred8 = predict(mdl8, X_test);
    [acc(i,8), sens(i,8), spec(i,8)] = get_metrics(y_test, pred8);
end

fprintf('                               ACC [%%]  Se [%%]  Sp [%%]\n');
fprintf('Fusion model based on final selection (6 features)\n');

names = {'SVM (Linear kernel)', 'SVM (Quadratic kernel)', 'SVM (Cubic kernel)', 'SVM (Gaussian kernel)', ...
         '1-nn (Euclidean Distance)', '1-nn (Chebyshev Distance)', '1-nn (Minkowski Distance)', '1-nn (Spearman Distance)'};

for m = 1:8
        fprintf('%-30s %-8.1f %-7.1f %-7.1f\n', names{m}, mean(acc(:,m))*100, mean(sens(:,m))*100, mean(spec(:,m))*100);
end

    fprintf('\nGenerating Confusion Matrix for Gaussian SVM...\n');
    
    % Train a cross-validated Gaussian SVM on the selected features
    mdl_best = fitcsvm(X, y, 'KernelFunction', 'gaussian', 'Standardize', true, 'CrossVal', 'on');
    pred_best = kfoldPredict(mdl_best);
    true_labels = categorical(y);
    predicted_labels = categorical(pred_best);
    figure('Name', 'Final Confusion Matrix', 'Color', 'w', 'Position', [200, 200, 650, 500]);
    cm = confusionchart(true_labels, predicted_labels);
    cm.Title = '10-Fold CV Confusion Matrix: Gaussian SVM';
    cm.RowSummary = 'row-normalized'; 
    cm.ColumnSummary = 'column-normalized'; 
    cm.DiagonalColor = [0.17 0.51 0.85]; 

function [acc, sens, spec] = get_metrics(truth, preds)
    cm = confusionmat(truth, preds);
    
    if size(cm,1) < 2 
        acc = sum(strcmp(truth,preds))/length(truth); sens=0; spec=0; return; 
    end
    
    TN = cm(1,1); FP = cm(1,2);
    FN = cm(2,1); TP = cm(2,2);
    
    acc = (TP+TN) / (TP+TN+FP+FN);
    sens = TP / (TP+FN);
    spec = TN / (TN+FP);
end
