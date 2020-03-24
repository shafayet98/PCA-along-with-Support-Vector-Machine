function X_rec = recoverData(Z, U, K)
X_rec = zeros(size(Z, 1), size(U, 1));            
    for i=1:size(Z, 1),
        for j=1:size(U,1),
            v = Z(i, :)';
            recovered_j = v' * U(j, 1:K)';
            X_rec(i, j) = recovered_j;
        end
    end
end
