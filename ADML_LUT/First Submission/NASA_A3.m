clear all
clc

%% First Import
dataset1 = load("train_FD001.txt");
dataset2 = load("train_FD002.txt");
dataset3 = load("train_FD003.txt");
dataset4 = load("train_FD004.txt");

%% ADML Nasa Turbo Fan A3

disp('===============================================================')
disp('Descriptive Statistics')
% Descriptive Statistics
% Calculate basic descriptive statistics for each dataset
mean_dataset1 = mean(dataset1);
median_dataset1 = median(dataset1);
std_dev_dataset1 = std(dataset1);
range_dataset1 = range(dataset1);

mean_dataset2 = mean(dataset2);
median_dataset2 = median(dataset2);
std_dev_dataset2 = std(dataset2);
range_dataset2 = range(dataset2);

mean_dataset3 = mean(dataset3);
median_dataset3 = median(dataset3);
std_dev_dataset3 = std(dataset3);
range_dataset3 = range(dataset3);

mean_dataset4 = mean(dataset4);
median_dataset4 = median(dataset4);
std_dev_dataset4 = std(dataset4);
range_dataset4 = range(dataset4);

% Detect outliers using the z-score method 
z_threshold = 3;

outliers_dataset1 = find(abs(zscore(dataset1)) > z_threshold);
outliers_dataset2 = find(abs(zscore(dataset2)) > z_threshold);
outliers_dataset3 = find(abs(zscore(dataset3)) > z_threshold);
outliers_dataset4 = find(abs(zscore(dataset4)) > z_threshold);

% Display the results
disp('Descriptive Statistics for Dataset 1:');
disp(['Mean: ', num2str(mean_dataset1)]);
disp(['Median: ', num2str(median_dataset1)]);
disp(['Standard Deviation: ', num2str(std_dev_dataset1)]);
disp(['Range: ', num2str(range_dataset1)]);
disp(['Number of Outliers: ', num2str(length(outliers_dataset1))]);

disp('Descriptive Statistics for Dataset 2:');
disp(['Mean: ', num2str(mean_dataset2)]);
disp(['Median: ', num2str(median_dataset2)]);
disp(['Standard Deviation: ', num2str(std_dev_dataset2)]);
disp(['Range: ', num2str(range_dataset2)]);
disp(['Number of Outliers: ', num2str(length(outliers_dataset2))]);

disp('Descriptive Statistics for Dataset 3:');
disp(['Mean: ', num2str(mean_dataset3)]);
disp(['Median: ', num2str(median_dataset3)]);
disp(['Standard Deviation: ', num2str(std_dev_dataset3)]);
disp(['Range: ', num2str(range_dataset3)]);
disp(['Number of Outliers: ', num2str(length(outliers_dataset3))]);

disp('Descriptive Statistics for Dataset 4:');
disp(['Mean: ', num2str(mean_dataset4)]);
disp(['Median: ', num2str(median_dataset4)]);
disp(['Standard Deviation: ', num2str(std_dev_dataset4)]);
disp(['Range: ', num2str(range_dataset4)]);
disp(['Number of Outliers: ', num2str(length(outliers_dataset4))]);

disp('===============================================================')
disp('Missing Data Analysis')
missing_values_dataset1 = sum(isnan(dataset1(:)));
missing_values_dataset2 = sum(isnan(dataset2(:)));
missing_values_dataset3 = sum(isnan(dataset3(:)));
missing_values_dataset4 = sum(isnan(dataset4(:)));

% Display the results
disp('Missing Values in Dataset 1:');
disp(['Number of Missing Values: ', num2str(missing_values_dataset1)]);
disp(['Percentage of Missing Values: ', num2str(100 * missing_values_dataset1 / numel(dataset1)), '%']);

disp('Missing Values in Dataset 2:');
disp(['Number of Missing Values: ', num2str(missing_values_dataset2)]);
disp(['Percentage of Missing Values: ', num2str(100 * missing_values_dataset2 / numel(dataset2)), '%']);

disp('Missing Values in Dataset 3:');
disp(['Number of Missing Values: ', num2str(missing_values_dataset3)]);
disp(['Percentage of Missing Values: ', num2str(100 * missing_values_dataset3 / numel(dataset3)), '%']);

disp('Missing Values in Dataset 4:');
disp(['Number of Missing Values: ', num2str(missing_values_dataset4)]);
disp(['Percentage of Missing Values: ', num2str(100 * missing_values_dataset4 / numel(dataset4)), '%']);

