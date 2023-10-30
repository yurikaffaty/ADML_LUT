clc
clear all;
close all;

%% import all of the files (train data, test data, RUL for test data)
model(1).dataset = load("train_FD001.txt");
model(2).dataset = load("train_FD002.txt");
model(3).dataset = load("train_FD003.txt");
model(4).dataset = load("train_FD004.txt");
model(1).testdata = load("test_FD001.txt");
model(2).testdata = load("test_FD002.txt");
model(3).testdata = load("test_FD003.txt");
model(4).testdata = load("test_FD004.txt");

model(1).RULT = load("RUL_FD001.txt");
model(2).RULT = load("RUL_FD002.txt");
model(3).RULT = load("RUL_FD003.txt");
model(4).RULT = load("RUL_FD004.txt");

% Model hyperparameters
% We want to calibrate with the healthy state of the process, where the cycles are low. 
% So we take the healthy beginning of the observations
calibration_ratio = 0.6; 

%% Calculating RUL : RUL(Observation) = maxtime(unit==currentunit) - time(Observation)

% Calculate RUL for training and test datasets
for i = 1:4  
    RUL_train{i} = cal_RUL_train(model(i).dataset);
    RUL_testdata{i} = cal_RUL_test(model(i).testdata, model(i).RULT);
end

% Rescale RUL for the entire dataset (training + test)
for i = 1:4
    combined_RUL = [RUL_train{i}; RUL_testdata{i}];
    rescaled_RUL = rescale(combined_RUL, 0, 1);
    RUL_train_rescaled{i} = rescaled_RUL(1:length(RUL_train{i}));
    RUL_test_rescaled{i} = rescaled_RUL(length(RUL_train{i})+1:end);
end

%% Preprocessing : Sort the rows, Split the data, and scale and center the data

figure; 

for i = 1:4  
    
    % Sorting the data
    % First we sort the rows, THEN we divide into calibration and validation. We want to calibrate with THE START OF THE PROCESS for each unit.
    % We want to calibrate with only the early cycles. that's why we sort the training dataset.
    [~, order] = sort(model(i).dataset(:,2)); 
    model(i).dataset = model(i).dataset(order,:);
    RUL_train_rescaled{i} = RUL_train_rescaled{i}(order);
  
    % Removing the first five columns because the operational setting will not change.
    model(i).dataset(:,  1:5)   = [];
    model(i).testdata(:, 1:5)   = [];

    
    % We splitted the data into calibration data (30%) and validation data lo(70%)
    nobs    = length(model(i).dataset(:,1));
    cal     = fix(nobs*0.3);
    idxCal  = logical([ones(cal, 1); zeros(nobs-cal, 1)]);
    idxVal  = logical([zeros(cal, 1); ones(nobs-cal, 1)]);
    model(i).calibration    = model(i).dataset(idxCal,:);
    model(i).validation     = model(i).dataset(idxVal,:);

    % Center and scale the data before PCA, using zscore
    [model(i).calC, model(i).mu, model(i).std] = zscore(model(i).calibration);
    model(i).valC = normalize(model(i).validation, 'center', model(i).mu, 'scale', model(i).std);
    model(i).valC(isnan(model(i).valC)) = 0;
    model(i).testC = normalize(model(i).testdata, 'center', model(i).mu, 'scale', model(i).std);
    model(i).testC(isnan(model(i).testC)) = 0;

    [model(i).P, model(i).T, ~, ~, model(i).explained] = pca(model(i).calC);

    nexttile; 
    plot(cumsum(model(i).explained));
    title('Cummulative explained variance by principal components, dataset' + string(i));
    xlabel('No. PCs in the model');
    ylabel('Explained variance of the model (%)');

end

% After we looked at the cummulative explained variance plots, we decided
% to select principal component 1 for dataset 1, 3 for dataset 2, 2 for
% dataset 3, and 3 for dataset 4.

decidedPCs = [1, 3, 2, 3];

%% Apply PCA, and compute T2 and SPEx

for i = 1:4
    figure;
    for j = 1:5

    decidedPCs(j) = j;
     % Calibrate model with selected no. PCs
    [model(i).P, model(i).T, model(i).latent, ~, model(i).explained] = pca(model(i).calC, "NumComponents", decidedPCs(i));

    % Calculate T2 and SPEx 
    model(i).SPExCal = qcomp(model(i).calC, model(i).P, decidedPCs(i));
    model(i).T2Cal   = t2comp(model(i).calC, model(i).P, model(i).latent, decidedPCs(i));

    model(i).SPExVal = qcomp(model(i).valC, model(i).P, decidedPCs(i));
    model(i).T2Val   = t2comp(model(i).valC, model(i).P, model(i).latent, decidedPCs(i));

    model(i).SPExTest = qcomp(model(i).testC, model(i).P, decidedPCs(i));
    model(i).T2Test   = t2comp(model(i).testC, model(i).P, model(i).latent, decidedPCs(i));
    
    % Calculate limits and warnings
    model(i).SPExAlert  = mean(model(i).SPExCal) + 3 * std(model(i).SPExCal);
    model(i).SPExWarning= mean(model(i).SPExCal) + 2 * std(model(i).SPExCal);

    model(i).T2Alert    = mean(model(i).T2Cal) + 3 * std(model(i).T2Cal);
    model(i).T2Warning  = mean(model(i).T2Cal) + 2 * std(model(i).T2Cal);

    pointsCal   = length(model(i).SPExCal); 
    pointsVal   = length(model(i).SPExVal);
    pointsTest  = length(model(i).SPExTest);
    noPoints    = pointsCal + pointsVal + pointsTest;

    nexttile;
    hold on
    scatter(1:pointsCal , model(i).SPExCal);
    scatter(pointsCal+1:pointsCal+pointsVal, model(i).SPExVal);
    scatter(pointsCal+pointsVal+1:pointsCal+pointsVal+pointsTest, model(i).SPExTest);
    plot([1 noPoints], [model(i).SPExWarning model(i).SPExWarning], '--');
    plot([1 noPoints], [model(i).SPExAlert model(i).SPExAlert], '--');
    title("SPEx control chart for dataset " + string(i) + " no PCs " + string(j));

    
    nexttile;
    hold on
    scatter(1:pointsCal , model(i).T2Cal);
    scatter(pointsCal+1:pointsCal+pointsVal, model(i).T2Val);
    scatter(pointsCal+pointsVal+1:pointsCal+pointsVal+pointsTest, model(i).T2Test);
    plot([1 noPoints], [model(i).T2Warning model(i).T2Warning], '--');
    plot([1 noPoints], [model(i).T2Alert model(i).T2Alert], '--');
    title("T^2 control chart for dataset " + string(i)+ " no PCs " + string(j));
    end
end

%% Plot RUL with gradient color

% Define the base color
A = [0, 1, 1];

% Find the maximum RUL value across datasets for the caption
max_RUL = max(cellfun(@max, RUL_train));

% An arbitrary large number assuming a dataset won't have more than 50 influential sensors.
max_sensors = 26; 
influential_sensor_T2 = NaN(4, max_sensors);
influential_sensor_SPEx = NaN(4, max_sensors);

% Plotting the control charts in the desired sequence
for i = 1:4

    % Number of points in each dataset partition
    pointsCal   = length(model(i).SPExCal); 
    pointsVal   = length(model(i).SPExVal);
    pointsTest  = length(model(i).SPExTest);
    noPoints    = pointsCal + pointsVal + pointsTest;

    % Color matrices for each partition
    colorCal = kron(A, ones(pointsCal,1)) .* rescale(RUL_train_rescaled{i}(1:pointsCal), 0, 1);
    colorVal = kron(A, ones(pointsVal,1)) .* rescale(RUL_train_rescaled{i}(pointsCal+1:end), 0, 1);
    colorTest = kron(A, ones(pointsTest,1)) .* RUL_test_rescaled{i};
    
    figure;

    % SPEx control chart
    subplot(2, 1, 1);
    hold on;
    scatter(1:pointsCal, model(i).SPExCal, 15, RUL_train_rescaled{i}(1:pointsCal), 'filled');
    scatter(pointsCal+1:pointsCal+pointsVal, model(i).SPExVal, 15, RUL_train_rescaled{i}(pointsCal+1:end), 'filled');
    scatter(pointsCal+pointsVal+1:noPoints, model(i).SPExTest, 15, RUL_test_rescaled{i}, 'filled');
    title("SPEx control chart for dataset " + string(i));
    xlabel('Observation Number');
    ylabel('SPEx Value');
    colorbar;
    annotation('textbox', [0.15,0.7,0.3,0.2], 'String', ['0 = RUL 1 and 1 = RUL ' num2str(max_RUL)], 'EdgeColor', 'none');
    
    % Signaling Incoming Failure within the SPEx subplot. We decided to try with cycle 10.
    signal_cycle = 10; 
    % Assuming RUL is rescaled between 0 to 1
    signal_idx_train = find(RUL_train_rescaled{i}(1:pointsCal) <= signal_cycle/100); 
    signal_idx_test = find(RUL_test_rescaled{i} <= signal_cycle/100);
    scatter(signal_idx_train, model(i).SPExCal(signal_idx_train), 25, 'r', 'filled', 'Marker', 's');
    scatter(pointsCal + pointsVal + signal_idx_test, model(i).SPExTest(signal_idx_test), 25, 'r', 'filled', 'Marker', 's');
   
    % T2 control chart
    subplot(2, 1, 2);
    hold on;
    scatter(1:pointsCal, model(i).T2Cal, 15, RUL_train_rescaled{i}(1:pointsCal), 'filled');
    scatter(pointsCal+1:pointsCal+pointsVal, model(i).T2Val, 15, RUL_train_rescaled{i}(pointsCal+1:end), 'filled');
    scatter(pointsCal+pointsVal+1:noPoints, model(i).T2Test, 15, RUL_test_rescaled{i}, 'filled');
    title("T^2 control chart for dataset " + string(i));
    xlabel('Observation Number');
    ylabel('T^2 Value');
    colorbar;
    annotation('textbox', [0.15,0.7,0.3,0.2], 'String', ['0 = RUL 1 and 1 = RUL ' num2str(max_RUL)], 'EdgeColor', 'none');

    % Signaling Incoming Failure within the T2 subplot
    signal_cycle = 10; % or 20
    signal_idx_train = find(RUL_train_rescaled{i}(1:pointsCal) <= signal_cycle/100); % Assuming RUL is rescaled between 0 to 1
    signal_idx_test = find(RUL_test_rescaled{i} <= signal_cycle/100);

    scatter(signal_idx_train, model(i).T2Cal(signal_idx_train), 25, 'r', 'filled', 'Marker', 's');
    scatter(pointsCal + pointsVal + signal_idx_test, model(i).T2Test(signal_idx_test), 25, 'r', 'filled', 'Marker', 's');
    
    % For T2
    out_of_control_idx_T2 = find(model(i).T2Val > model(i).T2Alert);
    disp(['Number of out-of-control observations in T^2 for dataset ' num2str(i) ': ' num2str(length(out_of_control_idx_T2))]);
    counter_T2 = 1;
    for idx = out_of_control_idx_T2'
        [~, influential_sensor] = max(abs(model(i).valC(idx, :)));
        influential_sensor_T2(i, counter_T2) = influential_sensor;
        counter_T2 = counter_T2 + 1;
    end
    
    % For SPEx
    out_of_control_idx_SPEx = find(model(i).SPExVal > model(i).SPExAlert);
    disp(['Number of out-of-control observations in SPEx for dataset ' num2str(i) ': ' num2str(length(out_of_control_idx_SPEx))]);

    counter_SPEx = 1;
    for idx = out_of_control_idx_SPEx'
        [~, influential_sensor] = max(abs(model(i).valC(idx, :) - (model(i).valC(idx, :) * model(i).P * model(i).P')));
        influential_sensor_SPEx(i, counter_SPEx) = influential_sensor;
        counter_SPEx = counter_SPEx + 1;
    end
    
end

% Display the results for the current dataset
for i = 1:4
    unique_T2_sensors = unique(influential_sensor_T2(i, ~isnan(influential_sensor_T2(i, :))));
    unique_SPEx_sensors = unique(influential_sensor_SPEx(i, ~isnan(influential_sensor_SPEx(i, :))));
    disp(['For dataset ' num2str(i) ', unique influential sensors in T^2 chart are: ' num2str(unique_T2_sensors)]);
    disp(['For dataset ' num2str(i) ', unique influential sensors in SPEx chart are: ' num2str(unique_SPEx_sensors)]);
    disp('------------------------------------------------------');
end

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

function T2     = t2comp(data, loadings, latent, comp)
score       = data * loadings(:,1:comp);
standscores = bsxfun(@times, score(:,1:comp), 1./sqrt(latent(1:comp,:))');
T2          = sum(standscores.^2,2);
end

function Qfac   = qcomp(data, loadings, comp)
score       = data * loadings(:,1:comp);
reconstructed = score * loadings(:,1:comp)';
residuals   = bsxfun(@minus, data, reconstructed);
Qfac        = sum(residuals.^2,2);
end
