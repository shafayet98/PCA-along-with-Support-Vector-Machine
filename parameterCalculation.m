function [C, sigma] = parameterCalculation(X, y, Xval, yval)

    % suppose c = 1 and sigma = 0.3

    C = 1;
    sigma = 0.3;

    results = eye(36,3);
    errorRow = 0;

    for C_test = [0.01 0.03 0.1 0.3 1, 3]
        for sigma_test = [0.01 0.03 0.1 0.3 1, 3]

            errorRow = errorRow + 1;
            model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
            predictions = svmPredict(model, Xval);
            prediction_error = mean(double(predictions ~= yval));

            results(errorRow,:) = [C_test, sigma_test, prediction_error];     
        end
    end

    sorted_results = sortrows(results, 3); % sort matrix by column-3, the error, ascending

    C = sorted_results(1,1);
    sigma = sorted_results(1,2);

end