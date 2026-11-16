%% benchmark_SIMD_conversion.m
% Full-dataset benchmark of RGB-to-LGB and RGB-to-YGB conversion.
% Prints results only; does not save output files.

clear;
clc;

datasetFolder = "C:\Users\DELL\Desktop\c\SIMD_RGB";
numRepeats = 5;

%% Discover all images
imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', ...
    {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'});

files = imds.Files;

if isempty(files)
    error("No images found. Check the SIMD dataset path: %s", ...
        datasetFolder);
end

fprintf("SIMD dataset: %s\n", datasetFolder);
fprintf("Image files found: %d\n", numel(files));

%% Load valid RGB images
imgs = cell(0,1);
skipped = 0;

for i = 1:numel(files)
    img = imread(files{i});

    if ndims(img) == 3 && size(img,3) == 3
        imgs{end+1,1} = img; %#ok<AGROW>
    else
        skipped = skipped + 1;
    end
end

N = numel(imgs);

if N == 0
    error("No valid three-channel RGB images found.");
end

fprintf("Valid RGB images: %d\n", N);
fprintf("Skipped non-RGB images: %d\n", skipped);
fprintf("Timing repetitions: %d\n\n", numRepeats);

%% Warm-up
rgb_to_lgb(imgs{1});
rgb_to_ygb(imgs{1});

timeLGB = zeros(numRepeats,1);
timeYGB = zeros(numRepeats,1);

%% Benchmark the full dataset
for r = 1:numRepeats
    fprintf("Run %d/%d...\n", r, numRepeats);

    if mod(r,2) == 1
        timeLGB(r) = measure_conversion(imgs, @rgb_to_lgb);
        timeYGB(r) = measure_conversion(imgs, @rgb_to_ygb);
    else
        timeYGB(r) = measure_conversion(imgs, @rgb_to_ygb);
        timeLGB(r) = measure_conversion(imgs, @rgb_to_lgb);
    end
end

%% Print results
lgb_ms = timeLGB * 1000 / N;
ygb_ms = timeYGB * 1000 / N;

fprintf("\n========== SIMD CONVERSION RESULTS ==========\n");
fprintf("Images processed: %d\n\n", N);

fprintf("LGB:\n");
fprintf("  Mean: %.4f ms/image\n", mean(lgb_ms));
fprintf("  Std:  %.4f ms/image\n", std(lgb_ms));
fprintf("  Mean full-dataset time: %.4f seconds\n\n", ...
    mean(timeLGB));

fprintf("YGB:\n");
fprintf("  Mean: %.4f ms/image\n", mean(ygb_ms));
fprintf("  Std:  %.4f ms/image\n", std(ygb_ms));
fprintf("  Mean full-dataset time: %.4f seconds\n\n", ...
    mean(timeYGB));

fprintf("LGB minus YGB: %.4f ms/image\n", ...
    mean(lgb_ms) - mean(ygb_ms));

fprintf("============================================\n");

%% Conversion-only timing
function elapsed = measure_conversion(imgs, converter)
    tic;
    for i = 1:numel(imgs)
        converted = converter(imgs{i}); %#ok<NASGU>
    end
    elapsed = toc;
end