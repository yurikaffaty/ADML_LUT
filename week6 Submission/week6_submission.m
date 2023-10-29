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
calibration_ratio = 0.6; % We want to calibrate with the healthy state of the process, where the cycles are low. So we take the healthy beginning of the observations

%% Calculating RUL : RUL(Observation) = maxtime(unit==currentunit) - time(Observation)
% NOTE! Thre is no need to add the RUL to the dataset. Also, the number of the unit and the cycle number is not a monitored parameter. Your code is very redundant, so I made it a bit more efficient 
% We can use the RUL in visualisation purposes

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


figure; 

for i = 1:4  
    

    % Sorting the data
    % First we sort the rows, THEN we divide into calibration and validation. We want to calibrate with THE START OF THE PROCESS for each unit.
    % We want to calibrate with only the early cycles.. that's why we sort
    % the training dataset.
    [~, order] = sort(model(i).dataset(:,2)); 
    model(i).dataset = model(i).dataset(order,:);
    RUL_train_rescaled{i} = RUL_train_rescaled{i}(order);
    % Removing the first two columns

    model(i).dataset(:,  1:5)   = [];
    model(i).testdata(:, 1:5)   = [];

    
    % Data in test doesn't need to be sorted
    nobs    = length(model(i).dataset(:,1));
    cal     = fix(nobs*0.3);
    idxCal  = logical([ones(cal, 1); zeros(nobs-cal, 1)]);
    idxVal  = logical([zeros(cal, 1); ones(nobs-cal, 1)]);
    model(i).calibration    = model(i).dataset(idxCal,:);
    model(i).validation     = model(i).dataset(idxVal,:);

    % Center and scale the data before PCA
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


decidedPCs = [1, 3, 2, 3];

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


% Define the base color
A = [0, 1, 1];

% Find the maximum RUL value across datasets for the caption
max_RUL = max(cellfun(@max, RUL_train));

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

    % T^2 control chart
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

