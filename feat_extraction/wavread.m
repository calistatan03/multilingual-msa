function [y, Fs] = wavread(filename, varargin)
% Minimal wavread compatibility shim for PCM WAV files.
% Supports the ffmpeg-generated files in this pipeline:
%   mono/stereo, PCM signed integer WAV, especially pcm_s16le.
%
% Usage:
%   [y, Fs] = wavread(filename)

    filename = char(filename);

    fid = fopen(filename, 'r', 'ieee-le');
    if fid == -1
        error('wavread:CannotOpenFile', 'Cannot open file: %s', filename);
    end
    cleaner = onCleanup(@() fclose(fid));

    % ---- RIFF header ----
    riff = fread(fid, 4, '*char')';
    if ~strcmp(riff, 'RIFF')
        error('wavread:InvalidFile', 'Not a RIFF file: %s', filename);
    end

    fread(fid, 1, 'uint32'); % file size
    wave = fread(fid, 4, '*char')';
    if ~strcmp(wave, 'WAVE')
        error('wavread:InvalidFile', 'Not a WAVE file: %s', filename);
    end

    % ---- scan chunks ----
    fmtFound = false;
    dataFound = false;

    numChannels = [];
    Fs = [];
    bitsPerSample = [];
    audioFormat = [];
    dataPos = [];
    dataSize = [];

    while ~feof(fid)
        chunkId = fread(fid, 4, '*char')';
        if numel(chunkId) < 4
            break;
        end
        chunkSize = fread(fid, 1, 'uint32');
        if isempty(chunkSize)
            break;
        end

        switch chunkId
            case 'fmt '
                fmtFound = true;
                audioFormat   = fread(fid, 1, 'uint16');   % 1 = PCM
                numChannels   = fread(fid, 1, 'uint16');
                Fs            = fread(fid, 1, 'uint32');
                fread(fid, 1, 'uint32'); % byteRate
                fread(fid, 1, 'uint16'); % blockAlign
                bitsPerSample = fread(fid, 1, 'uint16');

                remaining = double(chunkSize) - 16;
                if remaining > 0
                    fseek(fid, remaining, 'cof');
                end

            case 'data'
                dataFound = true;
                dataPos = ftell(fid);
                dataSize = chunkSize;
                fseek(fid, chunkSize, 'cof');

            otherwise
                % skip unknown chunk
                fseek(fid, chunkSize, 'cof');
        end

        % chunks are word-aligned
        if mod(chunkSize, 2) == 1
            fseek(fid, 1, 'cof');
        end
    end

    if ~fmtFound || ~dataFound
        error('wavread:MissingChunks', 'Missing fmt or data chunk in %s', filename);
    end

    if audioFormat ~= 1
        error('wavread:UnsupportedFormat', ...
            'Only PCM WAV supported. audioFormat=%d in %s', audioFormat, filename);
    end

    % ---- read audio data ----
    fseek(fid, dataPos, 'bof');

    switch bitsPerSample
        case 8
            raw = fread(fid, dataSize, 'uint8=>double');
            raw = (raw - 128) / 128;  % unsigned 8-bit PCM
        case 16
            raw = fread(fid, dataSize / 2, 'int16=>double');
            raw = raw / 32768;
        case 24
            % 24-bit PCM handling
            nSamples = dataSize / 3;
            b = fread(fid, [3, nSamples], 'uint8=>uint32')';
            rawInt = bitshift(b(:,3),16) + bitshift(b(:,2),8) + b(:,1);
            neg = rawInt >= 2^23;
            rawInt(neg) = rawInt(neg) - 2^24;
            raw = double(rawInt) / 2^23;
        case 32
            raw = fread(fid, dataSize / 4, 'int32=>double');
            raw = raw / 2147483648;
        otherwise
            error('wavread:UnsupportedBitDepth', ...
                'Unsupported bits per sample: %d in %s', bitsPerSample, filename);
    end

    if isempty(numChannels) || numChannels < 1
        error('wavread:BadChannels', 'Invalid channel count in %s', filename);
    end

    raw = reshape(raw, numChannels, [])';
    if numChannels == 1
        y = raw;
    else
        y = raw;
    end
end
