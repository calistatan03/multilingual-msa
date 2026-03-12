clear; clc;

% =========================
% USER PATHS
% =========================
covarep_root = '/hpctmp/scratch/e0968015/covarep';
project_root = '/home/svu/e0968015/feat_extraction';
data_root = '/hpctmp/scratch/e0968015/mosei';

label_csv = fullfile(data_root, 'label.csv');
wav_root  = fullfile(data_root, 'wav');
out_root  = fullfile(data_root, 'covarep_features');

% Add your own scripts folder so MATLAB can find
% extract_required_covarep_features.m
addpath(data_root);
addpath(project_root);

% =========================
% CHECKS
% =========================
if exist(label_csv, 'file') ~= 2
    error('label.csv not found: %s', label_csv);
end

if exist('extract_required_covarep_features', 'file') ~= 2
    error('extract_required_covarep_features.m not found on MATLAB path.');
end

if ~exist(out_root, 'dir')
    mkdir(out_root);
end

% =========================
% READ CSV
% =========================
opts = detectImportOptions(label_csv, 'TextType', 'string');

opts = setvartype(opts, 'video_id', 'string');
opts = setvartype(opts, 'clip_id', 'string');
opts = setvartype(opts, 'text', 'string');
opts = setvartype(opts, 'annotation', 'string');
opts = setvartype(opts, 'mode', 'string');

T = readtable(label_csv, opts);

class(T.video_id)
class(T.clip_id)
T(1:5, {'video_id','clip_id'})

required_cols = ["video_id", "clip_id"];
for k = 1:numel(required_cols)
    if ~ismember(required_cols(k), string(T.Properties.VariableNames))
        error('Missing required column in CSV: %s', required_cols(k));
    end
end

n = height(T);
fprintf('Found %d rows in %s\n', n, label_csv);

num_ok = 0;
num_skip = 0;
num_fail = 0;

% =========================
% MAIN LOOP
% =========================
for i = 1:n
    try
        video_id = safe_to_char(T.video_id(i));
        clip_id_raw = T.clip_id(i);

        if isnumeric(clip_id_raw)
            clip_id = sprintf('%d', clip_id_raw);
        else
            clip_id = strtrim(string(clip_id_raw));
        end

        audio_file = fullfile(wav_root, video_id, clip_id + ".wav");
        out_dir    = fullfile(out_root, video_id);
        out_mat    = fullfile(out_dir, clip_id + ".mat");

        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end

        if exist(audio_file, 'file') ~= 2
            fprintf(2, '[%d/%d] Missing WAV: %s\n', i, n, audio_file);
            num_fail = num_fail + 1;
            continue;
        end

        if exist(out_mat, 'file') == 2
            fprintf('[%d/%d] Skip existing: %s\n', i, n, out_mat);
            num_skip = num_skip + 1;
            continue;
        end

        fprintf('[%d/%d] Processing %s / %s\n', i, n, video_id, clip_id);

        extract_required_covarep_features(audio_file, out_mat, covarep_root);

        num_ok = num_ok + 1;

    catch ME
        fprintf(2, '[%d/%d] FAILED on %s / %s\n', ...
            i, n, string(T.video_id(i)), string(T.clip_id(i)));
        fprintf(2, 'Reason: %s\n', ME.message);

        for s = 1:numel(ME.stack)
            fprintf(2, '  at %s (line %d)\n', ME.stack(s).name, ME.stack(s).line);
        end

        num_fail = num_fail + 1;
    end
end

fprintf('\nDone.\n');
fprintf('Success: %d\n', num_ok);
fprintf('Skipped: %d\n', num_skip);
fprintf('Failed : %d\n', num_fail);

exit;
function out = safe_to_char(v)
% Convert table element into safe char for paths / fprintf

    if iscell(v)
        v = v{1};
    end

    if ismissing(v)
        out = '';
        return;
    end

    if isstring(v)
        if strlength(v) == 0
            out = '';
        else
            out = char(v);
        end
        return;
    end

    if ischar(v)
        out = v;
        return;
    end

    if isnumeric(v)
        if isempty(v) || isnan(v)
            out = '';
        else
            out = num2str(v);
        end
        return;
    end

    % fallback
    try
        out = char(string(v));
    catch
        out = '';
    end

end
