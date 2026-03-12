function extract_required_covarep_features(audio_file, out_mat, covarep_root)
% Extracts the required acoustic features for one WAV file:
%   - 12 MFCC-like cepstral coefficients via VAD_MFCC
%   - f0
%   - VUV
%   - glottal source parameters: NAQ, QOQ, H1H2, PSP, HRF
%   - MDQ
%   - peakSlope
%   - VAD
%
% Saves:
%   features      [T x 22]
%   feature_names [1 x 22 string]
%   time_vec      [T x 1]
%
% Note:
%   VAD is NOT MFCC. We compute MFCC separately using VAD_MFCC.

    addpath(genpath(covarep_root));

    % ---------- checks ----------
    assert(exist('COVAREP_feature_formant_extraction_perfile', 'file') == 2, ...
        'COVAREP_feature_formant_extraction_perfile not found.');
    assert(exist('VAD_MFCC', 'file') == 2, ...
        'VAD_MFCC not found.');

    % ---------- load audio ----------
    audio_file = char(audio_file);
    out_mat = char(out_mat)
    [x, fs] = audioread(audio_file);
    if size(x, 2) > 1
        x = mean(x, 2); % mono
    end

    % ---------- COVAREP features ----------
    opts = struct();
    opts.feature_fs = 0.01;  % 10 ms hop
    opts.features = { ...
        'f0', 'VUV', ...
        'NAQ', 'QOQ', 'H1H2', 'PSP', 'MDQ', 'HRF', ...
        'peakSlope', 'VAD' ...
    };
    opts.save_mat = false;
    opts.save_csv = false;

    results = COVAREP_feature_formant_extraction_perfile(audio_file, opts);

    covarep_names = {'f0','VUV','NAQ','QOQ','H1H2','PSP','MDQ','HRF','peakSlope','VAD'};

    % Build [T x 10] matrix from table
    covarep_mat = [];
    for i = 1:numel(covarep_names)
        if ~ismember(covarep_names{i}, results.Properties.VariableNames)
            error('Expected feature %s not found in COVAREP output.', covarep_names{i});
        end
        covarep_mat = [covarep_mat, results.(covarep_names{i})];
    end

    % ---------- MFCC-like cepstral coefficients ----------
    % VAD_MFCC returns [13 x T] in this repo implementation.
    % To get "12 coefficients", we drop the 0th coefficient and keep 2:13.
    mfcc_raw = VAD_MFCC(x, fs);   % [13 x T_mfcc]
    mfcc_raw = mfcc_raw';         % [T_mfcc x 13]

    if size(mfcc_raw, 2) < 13
        error('VAD_MFCC returned fewer than 13 coefficients.');
    end

    mfcc12 = mfcc_raw(:, 2:13);   % [T_mfcc x 12]

    % ---------- align frame counts ----------
    % Both are roughly 10 ms hop, but lengths may differ slightly.
    T_cov = size(covarep_mat, 1);
    T_mfc = size(mfcc12, 1);
    T_min = min(T_cov, T_mfc);

    covarep_mat = covarep_mat(1:T_min, :);
    mfcc12 = mfcc12(1:T_min, :);

    if ismember('time', results.Properties.VariableNames)
        time_vec = results.time(1:T_min);
    else
        time_vec = ((0:T_min-1)' * 0.01) + 0.005;
    end

    % ---------- final feature matrix ----------
    features = [mfcc12, covarep_mat];  % [T x 22]

    feature_names = [ ...
        "mfcc1","mfcc2","mfcc3","mfcc4","mfcc5","mfcc6", ...
        "mfcc7","mfcc8","mfcc9","mfcc10","mfcc11","mfcc12", ...
        "f0","VUV","NAQ","QOQ","H1H2","PSP","MDQ","HRF","peakSlope","VAD" ...
    ];

    save(out_mat, 'features', 'feature_names', 'time_vec', 'audio_file');
    [out_dir, ~, ~] = fileparts(out_mat);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
end
