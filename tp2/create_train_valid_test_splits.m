function [ XA XV XT ] = create_train_valid_test_splits( X )
    distribution = [0.7 0.15 0.15];
    distribution = floor(distribution * size(X,2));
    indices = randperm(size(X,2));
    
    e = cumsum(distribution);
    b = e - distribution + ones(1,size(distribution,2));
    
    XA = X(:, indices(b(1):e(1)));
    XV = X(:, indices(b(2):e(2)));
    XT = X(:, indices(b(3):e(3)));
end

