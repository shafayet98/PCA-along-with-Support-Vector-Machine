function [U, S] = pca(X_norm)
    [m, n] = size(X_norm);
    U = zeros(n);
    S = zeros(n);

    Sigma = 1.0/m .* X_norm' * X_norm;
    [U, S, V] = svd(Sigma);
end