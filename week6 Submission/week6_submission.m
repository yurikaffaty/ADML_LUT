clc
clear all;
close all;

%% import all of the files (train data, test data, RUL for test data)
dataset1 = load("train_FD001.txt");
dataset2 = load("train_FD002.txt");
dataset3 = load("train_FD003.txt");
dataset4 = load("train_FD004.txt");
testdata1 = load("test_FD001.txt");
testdata2 = load("test_FD002.txt");
testdata3 = load("test_FD003.txt");
testdata4 = load("test_FD004.txt");
RUL_test1 = load("RUL_FD001.txt");
RUL_test2 = load("RUL_FD002.txt");
RUL_test3 = load("RUL_FD003.txt");
RUL_test4 = load("RUL_FD004.txt");

%% Calculating RUL : RUL(Observation) = maxtime(unit==currentunit) - time(Observation)

% RUL for training data
RUL_traindata1 = cal_RUL_train(dataset1);
RUL_traindata2 = cal_RUL_train(dataset2);
RUL_traindata3 = cal_RUL_train(dataset3);
RUL_traindata4 = cal_RUL_train(dataset4);

% RUL for test data
RUL_testdata1 = cal_RUL_test(testdata1,RUL_test1);
RUL_testdata2 = cal_RUL_test(testdata2,RUL_test2);
RUL_testdata3 = cal_RUL_test(testdata3,RUL_test3);
RUL_testdata4 = cal_RUL_test(testdata4,RUL_test4);


%% Excluding columns 1-5 from datasets because the operational setting will not change.
% And we also add our RUL value at the end of dataset.
dataset1 = dataset1(:, 6:end);
dataset1 = [dataset1, RUL_traindata1];

dataset2 = dataset2(:, 6:end);
dataset2 = [dataset2, RUL_traindata2];

dataset3 = dataset3(:, 6:end);
dataset3 = [dataset3, RUL_traindata3];

dataset4 = dataset4(:, 6:end);
dataset4 = [dataset4, RUL_traindata4];

testdata1 = testdata1(:, 6:end);
testdata1= [testdata1, RUL_testdata1];

testdata2 = testdata2(:, 6:end);
testdata2= [testdata2, RUL_testdata2];

testdata3 = testdata3(:, 6:end);
testdata3= [testdata3, RUL_testdata3];

testdata4 = testdata4(:, 6:end);
testdata4= [testdata4, RUL_testdata4];

%% Checking missing values
missing_values1 = isnan(dataset1);
missing_count1 = sum(missing_values1);
any_missing1 = any(missing_values1(:));
missing_values2 = isnan(dataset2);
missing_count2 = sum(missing_values2);
any_missing2 = any(missing_values1(:));
missing_values3 = isnan(dataset3);
missing_count3 = sum(missing_values3);
any_missing3 = any(missing_values1(:));
missing_values4 = isnan(dataset4);
missing_count4 = sum(missing_values4);
any_missing4 = any(missing_values1(:));

%% Division of data into calibration (training) and validation
% We split the data into calibration for 80%, and validation for 20%
% Define the calibration ratio for each dataset (e.g., 0.8 for 80% calibration)
calibration_ratio = 0.2;

% Split each dataset into calibration and validation sets
[calibration_data1, validation_data1] = split_data(dataset1, calibration_ratio);
[calibration_data2, validation_data2] = split_data(dataset2, calibration_ratio);
[calibration_data3, validation_data3] = split_data(dataset3, calibration_ratio);
[calibration_data4, validation_data4] = split_data(dataset4, calibration_ratio);


%% PCA Model with Early-Cycle Observations

[Loading1, Scores1, EigenVals1, T2_1, Explained1, mu1] = pca(calibration_data1(:, 1:end-1));
[Loading2, Scores2, EigenVals2, T2_2, Explained2, mu2] = pca(calibration_data2(:, 1:end-1));
[Loading3, Scores3, EigenVals3, T2_3, Explained3, mu3] = pca(calibration_data3(:, 1:end-1));
[Loading4, Scores4, EigenVals4, T2_4, Explained4, mu4] = pca(calibration_data4(:, 1:end-1));

%% Visualize and comment the variation explained by the model with different no. PCs
figure;
plot(1:length(Explained1),cumsum(Explained1))
hold on
plot(1:length(Explained2),cumsum(Explained2))
plot(1:length(Explained3),cumsum(Explained3))
plot(1:length(Explained4),cumsum(Explained4))
title('Cummulative explained variance by principal components')
xlabel('No. PCs in the model')
ylabel('Explained variance of the model (%)')
legend('calibration data1','calibration data2','calibration data3','calibration data4')
hold off;
% From the plot, we will keep 4 principal components for dataset 1 because it explains
% most of variance at 99.8%.
% 3 principal components for dataset 2 and 4 because it explains most of variance at 99.42%.
% 5 principal components for dataset 3 because it explains most of variance at 99.85%.

%% Visualize the biplot of the principal components
% For dataset 1 : 2 components
ii = 1;
vbls = {'Sensor measurement1','Sensor measurement2','Sensor measurement3','Sensor measurement4','Sensor measurement5','Sensor measurement6','Sensor measurement7','Sensor measurement8','Sensor measurement9','Sensor measurement10','Sensor measurement11','Sensor measurement12','Sensor measurement13','Sensor measurement14','Sensor measurement15','Sensor measurement16','Sensor measurement17','Sensor measurement18','Sensor measurement19','Sensor measurement20','Sensor measurement21'};
figure;
for i = 1:3  
    subplot(2, 2, ii);
    biplot(Loading1(:,i:(i+1)), 'Scores', Scores1(:,i:(i+1)), 'VarLabels', vbls);
    text1 = "Component " + string(i);
    text2 = "Component " + string(i+1);
    xlabel(text1);
    ylabel(text2);  
    title('4 components for Dataset1')
    ii = ii + 1;
end

% For dataset 2 : 2 components
ii = 1;
figure;
for i = 1:2  
    subplot(2, 2, ii);
    biplot(Loading2(:,i:(i+1)), 'Scores', Scores2(:,i:(i+1)), 'VarLabels', vbls);
    text1 = "Component " + string(i);
    text2 = "Component " + string(i+1);
    xlabel(text1);
    ylabel(text2); 
    title('3 components for Dataset2')
    ii = ii + 1;
end

% For dataset 3 : 5 components
ii = 1;
figure;
for i = 1:4  
    subplot(3, 2, ii);
    biplot(Loading3(:,i:(i+1)), 'Scores', Scores3(:,i:(i+1)), 'VarLabels', vbls);
    text1 = "Component " + string(i);
    text2 = "Component " + string(i+1);
    xlabel(text1);
    ylabel(text2);
    title('5 components for Dataset3')
    ii = ii + 1;
end

% For dataset 4 : 3 components
ii = 1;
figure;
for i = 1:2  
    subplot(2, 2, ii);
    biplot(Loading4(:,i:(i+1)), 'Scores', Scores4(:,i:(i+1)), 'VarLabels', vbls);
    text1 = "Component " + string(i);
    text2 = "Component " + string(i+1);
    xlabel(text1);
    ylabel(text2); 
    title('3 components for Dataset4')
    ii = ii + 1;
end

%% Project the Late-Cycle Observations into the Model
projected_validation1 = (validation_data1(:, 1:end-1) - mu1) * Loading1;
projected_validation2 = (validation_data2(:, 1:end-1) - mu2) * Loading2;
projected_validation3 = (validation_data3(:, 1:end-1) - mu3) * Loading3;
projected_validation4 = (validation_data4(:, 1:end-1) - mu4) * Loading4;

% Calculate T^2 for projected validation data
T2_validation1 = arrayfun(@(i) projected_validation1(i,:) * inv(diag(EigenVals1)) * projected_validation1(i,:)', 1:size(projected_validation1, 1));
T2_validation2 = arrayfun(@(i) projected_validation2(i,:) * inv(diag(EigenVals2)) * projected_validation2(i,:)', 1:size(projected_validation2, 1));
T2_validation3 = arrayfun(@(i) projected_validation3(i,:) * inv(diag(EigenVals3)) * projected_validation3(i,:)', 1:size(projected_validation3, 1));
T2_validation4 = arrayfun(@(i) projected_validation4(i,:) * inv(diag(EigenVals4)) * projected_validation4(i,:)', 1:size(projected_validation4, 1));

% Transpose SPE_validation1 to match the dimensions of T2_validation1
reconstructed_validation1 = projected_validation1 * Loading1' + mu1;
SPE_validation1 = sum((validation_data1(:, 1:end-1) - reconstructed_validation1).^2, 2).';

reconstructed_validation2 = projected_validation2 * Loading2' + mu2;
SPE_validation2 = sum((validation_data2(:, 1:end-1) - reconstructed_validation2).^2, 2).';

reconstructed_validation3 = projected_validation3 * Loading3' + mu3;
SPE_validation3 = sum((validation_data3(:, 1:end-1) - reconstructed_validation3).^2, 2).';

reconstructed_validation4 = projected_validation4 * Loading4' + mu4;
SPE_validation4 = sum((validation_data4(:, 1:end-1) - reconstructed_validation4).^2, 2).';

%% Analyze Out-of-Control Samples

% Establish control limits (example: 95th percentile of T^2 and SPE from calibration data)
T2_limit1 = prctile(T2_validation1, 95);
SPE_limit1 = prctile(SPE_validation1, 95);

T2_limit2 = prctile(T2_validation2, 95);
SPE_limit2 = prctile(SPE_validation2, 95);

T2_limit3 = prctile(T2_validation3, 95);
SPE_limit3 = prctile(SPE_validation3, 95);

T2_limit4 = prctile(T2_validation4, 95);
SPE_limit4 = prctile(SPE_validation4, 95);

% Identify out-of-control observations in validation data
out_of_control_indices1 = find(T2_validation1 > T2_limit1 | SPE_validation1 > SPE_limit1);
out_of_control_indices2 = find(T2_validation2 > T2_limit2 | SPE_validation2 > SPE_limit2);
out_of_control_indices3 = find(T2_validation3 > T2_limit3 | SPE_validation3 > SPE_limit3);
out_of_control_indices4 = find(T2_validation4 > T2_limit4 | SPE_validation4 > SPE_limit4);

% Analyze contributions of out-of-control observations
contributions_validation1 = abs(validation_data1(out_of_control_indices1, 1:end-1) - reconstructed_validation1(out_of_control_indices1, :));
sensor_contributions1 = mean(contributions_validation1, 1);

contributions_validation2 = abs(validation_data2(out_of_control_indices2, 1:end-1) - reconstructed_validation2(out_of_control_indices2, :));
sensor_contributions2 = mean(contributions_validation2, 1);

contributions_validation3 = abs(validation_data3(out_of_control_indices3, 1:end-1) - reconstructed_validation3(out_of_control_indices3, :));
sensor_contributions3 = mean(contributions_validation3, 1);

contributions_validation4 = abs(validation_data4(out_of_control_indices4, 1:end-1) - reconstructed_validation4(out_of_control_indices4, :));
sensor_contributions4 = mean(contributions_validation4, 1);

% Sensors with higher values in sensor_contributions1 are the ones deviating more from the "normal" behavior established by the PCA model on the calibration data.

%% Test data
% Apply PCA into test data
projected_test1 = (testdata1(:, 1:end-1) - mu1) * Loading1;
projected_test2 = (testdata2(:, 1:end-1) - mu2) * Loading2;
projected_test3 = (testdata3(:, 1:end-1) - mu3) * Loading3;
projected_test4 = (testdata4(:, 1:end-1) - mu4) * Loading4;

% Calculate T^2 for projected test data
T2_test1 = arrayfun(@(i) projected_test1(i,:) * inv(diag(EigenVals1)) * projected_test1(i,:)', 1:size(projected_test1, 1));
T2_test2 = arrayfun(@(i) projected_test2(i,:) * inv(diag(EigenVals2)) * projected_test2(i,:)', 1:size(projected_test2, 1));
T2_test3 = arrayfun(@(i) projected_test3(i,:) * inv(diag(EigenVals3)) * projected_test3(i,:)', 1:size(projected_test3, 1));
T2_test4 = arrayfun(@(i) projected_test4(i,:) * inv(diag(EigenVals4)) * projected_test4(i,:)', 1:size(projected_test4, 1));

% Transpose SPE_test1 to match the dimensions of T2_test1
reconstructed_test1 = projected_test1 * Loading1' + mu1;
SPE_test1 = sum((testdata1(:, 1:end-1) - reconstructed_test1).^2, 2).';

reconstructed_test2 = projected_test2 * Loading2' + mu2;
SPE_test2 = sum((testdata2(:, 1:end-1) - reconstructed_test2).^2, 2).';

reconstructed_test3 = projected_test3 * Loading3' + mu3;
SPE_test3 = sum((testdata3(:, 1:end-1) - reconstructed_test3).^2, 2).';

reconstructed_test4 = projected_test4 * Loading4' + mu4;
SPE_test4 = sum((testdata4(:, 1:end-1) - reconstructed_test4).^2, 2).';

% Establish control limits for test data (using the 95th percentile from calibration data)
T2_limit_test1 = prctile(T2_validation1, 95);
SPE_limit_test1 = prctile(SPE_validation1, 95);

T2_limit_test2 = prctile(T2_validation2, 95);
SPE_limit_test2 = prctile(SPE_validation2, 95);

T2_limit_test3 = prctile(T2_validation3, 95);
SPE_limit_test3 = prctile(SPE_validation3, 95);

T2_limit_test4 = prctile(T2_validation4, 95);
SPE_limit_test4 = prctile(SPE_validation4, 95);

% Identify out-of-control observations in test data
out_of_control_indices_test1 = find(T2_test1 > T2_limit_test1 | SPE_test1 > SPE_limit_test1);
out_of_control_indices_test2 = find(T2_test2 > T2_limit_test2 | SPE_test2 > SPE_limit_test2);
out_of_control_indices_test3 = find(T2_test3 > T2_limit_test3 | SPE_test3 > SPE_limit_test3);
out_of_control_indices_test4 = find(T2_test4 > T2_limit_test4 | SPE_test4 > SPE_limit_test4);

% Calculate T^2 and SPE for the training data
T2_train1 = arrayfun(@(i) projected_validation1(i, :) * inv(diag(EigenVals1)) * projected_validation1(i, :)', 1:size(projected_validation1, 1));
SPE_train1 = sum((validation_data1(:, 1:end-1) - reconstructed_validation1).^2, 2).';

T2_train2 = arrayfun(@(i) projected_validation2(i, :) * inv(diag(EigenVals2)) * projected_validation2(i, :)', 1:size(projected_validation2, 1));
SPE_train2 = sum((validation_data2(:, 1:end-1) - reconstructed_validation2).^2, 2).';

T2_train3 = arrayfun(@(i) projected_validation3(i, :) * inv(diag(EigenVals3)) * projected_validation3(i, :)', 1:size(projected_validation3, 1));
SPE_train3 = sum((validation_data3(:, 1:end-1) - reconstructed_validation3).^2, 2).';

T2_train4 = arrayfun(@(i) projected_validation4(i, :) * inv(diag(EigenVals4)) * projected_validation4(i, :)', 1:size(projected_validation4, 1));
SPE_train4 = sum((validation_data4(:, 1:end-1) - reconstructed_validation4).^2, 2).';

% Transpose SPE_train1 to match the dimensions of T2_train1
reconstructed_train1 = projected_validation1 * Loading1' + mu1;
SPE_train1 = sum((validation_data1(:, 1:end-1) - reconstructed_train1).^2, 2).';

reconstructed_train2 = projected_validation2 * Loading2' + mu2;
SPE_train2 = sum((validation_data2(:, 1:end-1) - reconstructed_train2).^2, 2).';

reconstructed_train3 = projected_validation3 * Loading3' + mu3;
SPE_train3 = sum((validation_data3(:, 1:end-1) - reconstructed_train3).^2, 2).';

reconstructed_train4 = projected_validation4 * Loading4' + mu4;
SPE_train4 = sum((validation_data4(:, 1:end-1) - reconstructed_train4).^2, 2).';

% Establish control limits for the training data
T2_limit_train1 = prctile(T2_train1, 95);
SPE_limit_train1 = prctile(SPE_train1, 95);

T2_limit_train2 = prctile(T2_train2, 95);
SPE_limit_train2 = prctile(SPE_train2, 95);

T2_limit_train3 = prctile(T2_train3, 95);
SPE_limit_train3 = prctile(SPE_train3, 95);

T2_limit_train4 = prctile(T2_train4, 95);
SPE_limit_train4 = prctile(SPE_train4, 95);

% Identify out-of-control observations in training data
out_of_control_indices_train1 = find(T2_train1 > T2_limit_train1 | SPE_train1 > SPE_limit_train1);
out_of_control_indices_train2 = find(T2_train2 > T2_limit_train2 | SPE_train2 > SPE_limit_train2);
out_of_control_indices_train3 = find(T2_train3 > T2_limit_train3 | SPE_train3 > SPE_limit_train3);
out_of_control_indices_train4 = find(T2_train4 > T2_limit_train4 | SPE_train4 > SPE_limit_train4);

% Plot control charts for Training Dataset 1
plot_control_charts(T2_train1, SPE_train1, T2_limit_train1, SPE_limit_train1, out_of_control_indices_train1, 'Training Dataset 1');
% Plot control charts for Training Dataset 2
plot_control_charts(T2_train2, SPE_train2, T2_limit_train2, SPE_limit_train2, out_of_control_indices_train2, 'Training Dataset 2');
% Plot control charts for Training Dataset 3
plot_control_charts(T2_train3, SPE_train3, T2_limit_train3, SPE_limit_train3, out_of_control_indices_train3, 'Training Dataset 3');
% Plot control charts for Training Dataset 4
plot_control_charts(T2_train4, SPE_train4, T2_limit_train4, SPE_limit_train4, out_of_control_indices_train4, 'Training Dataset 4');

% Plot control charts for Test Dataset 1
plot_control_charts(T2_test1, SPE_test1, T2_limit_test1, SPE_limit_test1, out_of_control_indices_test1, 'Test Dataset 1');
% Plot control charts for Test Dataset 2
plot_control_charts(T2_test2, SPE_test2, T2_limit_test2, SPE_limit_test2, out_of_control_indices_test2, 'Test Dataset 2');
% Plot control charts for Test Dataset 3
plot_control_charts(T2_test3, SPE_test3, T2_limit_test3, SPE_limit_test3, out_of_control_indices_test3, 'Test Dataset 3');
% Plot control charts for Test Dataset 4
plot_control_charts(T2_test4, SPE_test4, T2_limit_test4, SPE_limit_test4, out_of_control_indices_test4, 'Test Dataset 4');

%% Function
function RUL_train = cal_RUL_train(dataset)
    unique_units = unique(dataset(:, 1));
    num_units = length(unique_units);

    RUL_train = zeros(size(dataset, 1), 1);

    for i = 1:num_units
        unit_idx = dataset(:, 1) == unique_units(i);
        num_observations = sum(unit_idx);
        RUL_train(unit_idx) = num_observations:-1:1;
    end
end

function RUL_testdata = cal_RUL_test(testdata, RUL_test)
    unique_units = unique(testdata(:, 1));  
    num_units = length(unique_units);       
    RUL_testdata = zeros(size(testdata, 1), 1);

    for i = 1:num_units
        unit_idx = testdata(:, 1) == unique_units(i);
        num_observations = sum(unit_idx);
        RUL_values = RUL_test(i); 
        RUL_testdata(unit_idx) = RUL_values + (num_observations:-1:1)';
    end
end

function [calibration_data, validation_data] = split_data(dataset, calibration_ratio)
    % Determine the number of observations in the dataset
    num_observations = size(dataset, 1);
    
    % Create a cvpartition object for the specified calibration ratio
    c = cvpartition(num_observations, 'HoldOut', calibration_ratio);
    
    % Generate the logical index for the calibration and validation sets
    calibration_idx = training(c);
    validation_idx = test(c);
    
    % Split the data into calibration and validation sets
    calibration_data = dataset(calibration_idx, :);
    validation_data = dataset(validation_idx, :);
end

function plot_control_charts(T2, SPE, T2_limit, SPE_limit, out_of_control_indices, title_str)
    figure;
    n = length(T2);
    subplot(2, 1, 1);
    
    % Plot T2 chart
    plot(1:n, T2, 'b', 'LineWidth', 1.5);
    hold on;
    
    % Check if out_of_control_indices is empty to avoid errors
    if ~isempty(out_of_control_indices)
        plot(out_of_control_indices, T2(out_of_control_indices), 'r.', 'MarkerSize', 6);
    end
    
    plot([1, n], [T2_limit, T2_limit], 'k--', 'LineWidth', 1.5);
    
    title(['T2 Control Chart - ' title_str]);
    xlabel('Sample Index');
    ylabel('T2 Value');
    legend('T2', 'Out-of-Control Samples', '95% Control Limit', 'Location', 'Best');
    grid on;
    
    subplot(2, 1, 2);
    
    % Plot SPE chart
    plot(1:n, SPE, 'g', 'LineWidth', 1.5);
    hold on;
    
    % Check if out_of_control_indices is empty to avoid errors
    if ~isempty(out_of_control_indices)
        plot(out_of_control_indices, SPE(out_of_control_indices), 'r.', 'MarkerSize', 6);
    end
    
    plot([1, n], [SPE_limit, SPE_limit], 'k--', 'LineWidth', 1.5);
    
    title(['SPE Control Chart - ' title_str]);
    xlabel('Sample Index');
    ylabel('SPE Value');
    legend('SPE', 'Out-of-Control Samples', '95% Control Limit', 'Location', 'Best');
    grid on;
end

