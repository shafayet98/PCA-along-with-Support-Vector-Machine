function Z = projectData(X, U, K)
    Z = zeros(size(X, 1), K);
    for i=1:size(X, 1),
        for j=1:K,
            x = X(i, :)';
            projection_k = x' * U(:, j);
            Z(i, j) = projection_k;
        end
    end
end