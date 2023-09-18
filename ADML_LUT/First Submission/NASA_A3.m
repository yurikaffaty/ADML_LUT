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

% Visualization Dataset
% Boxplot for distribution and Outliers
% First, we use zscore (stadardization) for uniform scale of data

scaled_dataset1 = zscore(dataset1(:,3:26));
scaled_dataset2 = zscore(dataset2(:,3:26));
scaled_dataset3 = zscore(dataset3(:,3:26));
scaled_dataset4 = zscore(dataset4(:,3:26));

figure(1);
subplot(2,2,1)
boxplot(scaled_dataset1);
xlabel('Variables')
ylabel('Data')
title('Boxplot for train FD001')

subplot(2,2,2)
boxplot(scaled_dataset2);
xlabel('Variables')
ylabel('Data')
title('Boxplot for train FD002')

subplot(2,2,3)
boxplot(scaled_dataset3);
xlabel('Variables')
ylabel('Data')
title('Boxplot for train FD003')

subplot(2,2,4)
boxplot(scaled_dataset4);
xlabel('Variables')
ylabel('Data')
title('Boxplot for train FD004')

% Using histogram to visualize the distribution of each variable.

figure(2);
plothistogram(dataset1, 1, 'Histogram for train FD001');
plothistogram(dataset2, 2, 'Histogram for train FD002');
plothistogram(dataset3, 3, 'Histogram for train FD003');
plothistogram(dataset4, 4, 'Histogram for train FD004');


% Time series plot for measurements over time
% Number of sensor measurement (from column 6 to 26)
no_sensor_measurement = 21; 

figure(3);

for i = 1:no_sensor_measurement
    subplot(5, 5, i); 
    plot(dataset1(:,2), dataset1(:, i+5));
    xlabel('Time, in cycles');
    ylabel('Sensor Measurement Value');
    title(['Senser Measurement ' num2str(i)]);
end

figure(4);
for i = 1:no_sensor_measurement
    subplot(5, 5, i);  
    plot(dataset2(:,2), dataset2(:, i+5));
    xlabel('Time, in cycles');
    ylabel('Sensor Measurement Value');
    title(['Senser Measurement ' num2str(i)]);
end

figure(5);
for i = 1:no_sensor_measurement
    subplot(5, 5, i); 
    plot(dataset3(:,2), dataset3(:, i+5));
    xlabel('Time, in cycles');
    ylabel('Sensor Measurement Value');
    title(['Senser Measurement ' num2str(i)]);
end

figure(6);
for i = 1:no_sensor_measurement
    subplot(5, 5, i); 
    plot(dataset4(:,2), dataset4(:, i+5));
    xlabel('Time, in cycles');
    ylabel('Sensor Measurement Value');
    title(['Senser Measurement ' num2str(i)]);
end

% Function for plot histogram
function plothistogram(dataset, position, titleName)
    subplot(2, 2, position)
    for i = 3:26
        histogram(dataset(:,i));
    end
    title(titleName);
end
