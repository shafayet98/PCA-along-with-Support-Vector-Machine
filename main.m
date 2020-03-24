% 3 parts: Principal Component Analysis

% 1. visualize the data
% 2. project data (reduce the dimention)
% 3. recover data

data = csvread("dataset.csv");
data = data(2:length(data),1:14);
data = data(randperm(size(data, 1)), :); % randomly shuffling rows in matrix 

X = data(:,1:13);
y = data(:,14);

[X_norm, mu, sigma] = normalization(X);

%  Run PCA (by running sigular value decomposition)
[U, S] = pca(X_norm);

% Dimension Reduction
% Project the data onto K = 2 dimension
K = 2;
Z = projectData(X_norm, U, K); % 303X2

% plot data with reduced feature
Z = [Z y]; 
plot(Z(Z(:,3)==1, 1), Z(Z(:,3)==1, 2), 'k.');
hold on;
plot(Z(Z(:,3)==0, 1), Z(Z(:,3)==0, 2), 'r.');
% disp(size(Z)); 303X3

% Recover data
Z  = Z(:,1:2);
X_rec  = recoverData(Z, U, K);
% disp(size(X_rec)); 303X13
% disp(X_rec);

% Support Vector Machine Implementation
X_train = X_norm(1:200,1:end);
y_train = y(1:200,1:end);

X_cv = X_norm(201:250,1:end);
y_cv = y(201:250,1:end);

X_test = X_norm(251:303,1:end);
y_test = y(251:303,1:end);

% % Calculate the different SVM parameter and take the one gives lowest cost
[C, sigma] = parameterCalculation(X_train, y_train, X_cv, y_cv);

% % Train the SVM with the C and sigma we got
model = svmTrain(X_train, y_train, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, X_cv);

PredictionVsActual = [predictions y_cv];
disp(PredictionVsActual);
% accurecy meserment
fprintf('Train Accuracy: %f\n', mean(double(predictions == y_cv)) * 100);