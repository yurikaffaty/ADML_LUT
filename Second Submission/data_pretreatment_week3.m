clc
clear all;
close all;

dataset1 = load("train_FD001.txt");
dataset2 = load("train_FD002.txt");
dataset3 = load("train_FD003.txt");
dataset4 = load("train_FD004.txt");

%% Calculating RUL : RUL(Observation) = maxtime(unit==currentunit) - time(Observation)

% RUL for dataset 1
unique_units1 = unique(dataset1(:,1));
RUL_dataset1 = zeros(size(dataset1, 1), 1);
for unit = unique_units1'
    idx = dataset1(:,1) == unit;
    max_cycle_for_unit = max(dataset1(idx,2));
    RUL_dataset1(idx) = max_cycle_for_unit - dataset1(idx,2);
end

% RUL for dataset2
unique_units2 = unique(dataset2(:,1));
RUL_dataset2 = zeros(size(dataset2, 1), 1);
for unit = unique_units2'
    idx = dataset2(:,1) == unit;
    max_cycle_for_unit = max(dataset2(idx,2));
    RUL_dataset2(idx) = max_cycle_for_unit - dataset2(idx,2);
end

% RUL for dataset3
unique_units3 = unique(dataset3(:,1));
RUL_dataset3 = zeros(size(dataset3, 1), 1);
for unit = unique_units3'
    idx = dataset3(:,1) == unit;
    max_cycle_for_unit = max(dataset3(idx,2));
    RUL_dataset3(idx) = max_cycle_for_unit - dataset3(idx,2);
end

% RUL for dataset4
unique_units4 = unique(dataset4(:,1));
RUL_dataset4 = zeros(size(dataset4, 1), 1);
for unit = unique_units4'
    idx = dataset4(:,1) == unit;
    max_cycle_for_unit = max(dataset4(idx,2));
    RUL_dataset4(idx) = max_cycle_for_unit - dataset4(idx,2);
end

%% Excluding columns 1-5 from datasets because the operational setting will not change.
% And we also add our RUL value at the end of dataset.
dataset1 = dataset1(:, 6:end);
dataset1 = [dataset1, RUL_dataset1];

dataset2 = dataset2(:, 6:end);
dataset2 = [dataset2, RUL_dataset2];

dataset3 = dataset3(:, 6:end);
dataset3 = [dataset3, RUL_dataset3];

dataset4 = dataset4(:, 6:end);
dataset4 = [dataset4, RUL_dataset4];


%% Division of data into calibration (training), validation and test partitions
% We split the data into calibration for 30%, validation and test data for
% 35% per each. 
calibration_split = 0.3;
validation_split = 0.35;

[calibration_data1, validation_data1, test_data1] = split_data(dataset1, calibration_split, validation_split);
[calibration_data2, validation_data2, test_data2] = split_data(dataset2, calibration_split, validation_split);
[calibration_data3, validation_data3, test_data3] = split_data(dataset3, calibration_split, validation_split);
[calibration_data4, validation_data4, test_data4] = split_data(dataset4, calibration_split, validation_split);


%% Data centering and scaling techniques
% Using zscore to center and scale the data
[calibration_data1, mean1, std1] = zscore(calibration_data1);
[calibration_data2, mean2, std2] = zscore(calibration_data2);
[calibration_data3, mean3, std3] = zscore(calibration_data3);
[calibration_data4, mean4, std4] = zscore(calibration_data4);

% Using mean and std from training data to normalize validation and test
% data
validation_data1 = normalize_data(validation_data1, mean1, std1);
test_data1 = normalize_data(test_data1, mean1, std1);
validation_data2 = normalize_data(validation_data2, mean2, std2);
test_data2 = normalize_data(test_data2, mean2, std2);
validation_data3 = normalize_data(validation_data3, mean3, std3);
test_data3 = normalize_data(test_data3, mean3, std3);
validation_data4 = normalize_data(validation_data4, mean4, std4);
test_data4 = normalize_data(test_data4, mean4, std4);

% Checking missing values. Fill NaN with Zeros
test_data1(isnan(test_data1)) = 0;
test_data2(isnan(test_data2)) = 0;
test_data3(isnan(test_data3)) = 0;
test_data4(isnan(test_data4)) = 0;


%% Apply PCA on the model. Treat all variables as dependent (input) variables.
[Loading1, Scores1, EigenVals1, T2_1, Explained1, mu1] = pca(calibration_data1);
[Loading2, Scores2, EigenVals2, T2_2, Explained2, mu2] = pca(calibration_data2);
[Loading3, Scores3, EigenVals3, T2_3, Explained3, mu3] = pca(calibration_data3);
[Loading4, Scores4, EigenVals4, T2_4, Explained4, mu4] = pca(calibration_data4);
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
legend('calibration_data1','calibration_data2','calibration_data3','calibration_data4')
hold off;
% From the plot, we will keep 15 principal components because it explains
% most of variance at 99%.

%% Compute the biplot of the principal components
% For dataset 1 : 8 components
ii = 1;
vbls = {'Sensor measurement1','Sensor measurement2','Sensor measurement3','Sensor measurement4','Sensor measurement5','Sensor measurement6','Sensor measurement7','Sensor measurement8','Sensor measurement9','Sensor measurement10','Sensor measurement11','Sensor measurement12','Sensor measurement13','Sensor measurement14','Sensor measurement15','Sensor measurement16','Sensor measurement17','Sensor measurement18','Sensor measurement19','Sensor measurement20','Sensor measurement21','RUL'};
figure;
for i = 1:7  
    subplot(4, 2, ii);
    biplot(Loading1(:,i:(i+1)), 'Scores', Scores1(:,i:(i+1)), 'VarLabels', vbls);
    text1 = "Component " + string(i);
    text2 = "Component " + string(i+1);
    xlabel(text1);
    ylabel(text2);  
    title('8 components for Dataset1')
    ii = ii + 1;
end


% For dataset 2 : 4 components
ii = 1;
figure;
for i = 1:3  
    subplot(2, 2, ii);
    biplot(Loading2(:,i:(i+1)), 'Scores', Scores2(:,i:(i+1)), 'VarLabels', vbls);
    text1 = "Component " + string(i);
    text2 = "Component " + string(i+1);
    xlabel(text1);
    ylabel(text2); 
    title('4 components for Dataset2')
    ii = ii + 1;
end


% For dataset 3 : 6 components
ii = 1;
figure;
for i = 1:5  
    subplot(3, 2, ii);
    biplot(Loading3(:,i:(i+1)), 'Scores', Scores3(:,i:(i+1)), 'VarLabels', vbls);
    text1 = "Component " + string(i);
    text2 = "Component " + string(i+1);
    xlabel(text1);
    ylabel(text2);
    title('6 components for Dataset3')
    ii = ii + 1;
end


% For dataset 4 : 4 components
ii = 1;
figure;
for i = 1:3  
    subplot(2, 2, ii);
    biplot(Loading4(:,i:(i+1)), 'Scores', Scores4(:,i:(i+1)), 'VarLabels', vbls);
    text1 = "Component " + string(i);
    text2 = "Component " + string(i+1);
    xlabel(text1);
    ylabel(text2); 
    title('4 components for Dataset4')
    ii = ii + 1;
end

%% Compute T2 and SPE for dataset 1

% PCA results stored into a structured variable
model(1).X = calibration_data1;
model(1).P = Loading1;
model(1).latent = EigenVals1;

model(2).X = calibration_data2;
model(2).P = Loading2;
model(2).latent = EigenVals2;

model(3).X = calibration_data3;
model(3).P = Loading3;
model(3).latent = EigenVals3;

model(4).X = calibration_data4;
model(4).P = Loading4;
model(4).latent = EigenVals4;

% Number of PCs to consider (assuming 15 as mentioned)
nPCs = [8 4 6 4];

for k = 1:4  % Loop through each dataset
    contrT2 = t2contr(model(k).X, model(k).P, model(k).latent, nPCs(k));
    contrQ  = qcontr(model(k).X, model(k).P, nPCs(k));

    % Plotting T2 contribution
    figure;
    bar(contrT2);
    title("Variable Contributions to T2 for Dataset " + k);

    % Plotting SPEx contribution
    figure;
    bar(contrQ);
    title("Variable Contributions to SPEx for Dataset " + k);
end

%% 
alpha = 0.05; % significance level for control limits

% Function to compute T2 statistic for given data, loadings, and latent values
computeT2 = @(X,P,latent,nPCs) sum((X * P(:,1:nPCs)).^2 ./ latent(1:nPCs)', 2);

% Function to compute SPE statistic
computeSPE = @(X,P,nPCs) sum((X - X * P(:,1:nPCs) * P(:,1:nPCs)').^2, 2);

for k = 1:4  % Loop through each dataset
    % Compute T2 and SPE for calibration data
    T2_cal = computeT2(model(k).X, model(k).P, model(k).latent, nPCs(k));
    SPE_cal = computeSPE(model(k).X, model(k).P, nPCs(k));
    
    % Set control limits
    T2_limit = quantile(T2_cal, 1-alpha);
    SPE_limit = quantile(SPE_cal, 1-alpha);
    
    % Compute T2 and SPE for test data
    if k == 1
        test_data = test_data1;
    elseif k == 2
        test_data = test_data2;
    elseif k == 3
        test_data = test_data3;
    else
        test_data = test_data4;
    end
    
    T2_test = computeT2(test_data, model(k).P, model(k).latent, nPCs(k));
    SPE_test = computeSPE(test_data, model(k).P, nPCs(k));
    
    % Plot control charts for T2
    figure;
    plot(T2_cal, 'b');
    hold on;
    plot(length(T2_cal) + (1:length(T2_test)), T2_test, 'r');
    plot([1, length(T2_cal)+length(T2_test)], [T2_limit, T2_limit], 'k--');
    title(['T^2 Control Chart for Dataset ', num2str(k)]);
    legend('Calibration Data', 'Test Data', 'Control Limit');
    hold off;

    % Plot control charts for SPE
    figure;
    plot(SPE_cal, 'b');
    hold on;
    plot(length(SPE_cal) + (1:length(SPE_test)), SPE_test, 'r');
    plot([1, length(SPE_cal)+length(SPE_test)], [SPE_limit, SPE_limit], 'k--');
    title(['SPE Control Chart for Dataset ', num2str(k)]);
    legend('Calibration Data', 'Test Data', 'Control Limit');
    hold off;
end

% To determine the RUL when the test data goes out of control, simply find the first instance where the T2 or SPE exceeds the limit.
for k = 1:4
    if k == 1
        RUL_values = dataset1(:, end);
    elseif k == 2
        RUL_values = dataset2(:, end);
    elseif k == 3
        RUL_values = dataset3(:, end);
    else
        RUL_values = dataset4(:, end);
    end
    
    out_of_control_idx = find(T2_test > T2_limit | SPE_test > SPE_limit, 1);
    if ~isempty(out_of_control_idx)
        disp(['Dataset ', num2str(k), ' goes out of control at RUL: ', num2str(RUL_values(out_of_control_idx))]);
    else
        disp(['Dataset ', num2str(k), ' never goes out of control']);
    end
end

%% Function part
function [calibration_data, validation_data, test_data] = split_data(data, calibration_ratio, validation_ratio)
    total_rows = size(data, 1);

    % Determine the number of rows for each split based on the provided ratios
    calibration_rows = floor(total_rows * calibration_ratio);
    validation_rows = floor(total_rows * validation_ratio);
    test_rows = total_rows - calibration_rows - validation_rows;

    % Split the dataset
    calibration_data = data(1:calibration_rows, :);
    validation_data = data(calibration_rows+1:calibration_rows+validation_rows, :);
    test_data = data(calibration_rows+validation_rows+1:end, :);
end

% normalization function
function normalized_data = normalize_data(data, mean_val, std_val)
    normalized_data = (data - mean_val) ./ std_val;
end

function T2varcontr = t2contr(data, loadings, latent, comp)
    score = data * loadings(:,1:comp);
    standscores = bsxfun(@times, score(:,1:comp), 1./sqrt(latent(1:comp,:))');
    T2contr = abs(standscores*loadings(:,1:comp)');
    T2varcontr = sum(T2contr,1);
end

function Qcontr = qcontr(data, loadings, comp)
    score = data * loadings(:,1:comp);
    reconstructed = score * loadings(:,1:comp)';
    residuals = bsxfun(@minus, data, reconstructed);
    Qcontr = sum(residuals.^2);
end
