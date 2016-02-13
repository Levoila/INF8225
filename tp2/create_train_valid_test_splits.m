function [ XA XV XT YA YV YT ] = create_train_valid_test_splits( X, Y )
    distribution = [0.7 0.15 0.15];
    distribution = floor(distribution * size(X,2));
    indices = randperm(size(X,2));
    
    e = cumsum(distribution);
    b = e - distribution + ones(1,size(distribution,2));
    
    XA = X(:, indices(b(1):e(1)));
    XV = X(:, indices(b(2):e(2)));
    XT = X(:, indices(b(3):e(3)));
    
    YA = Y(indices(b(1):e(1)),:);
    YV = Y(indices(b(2):e(2)),:);
    YT = Y(indices(b(3):e(3)),:);
end

