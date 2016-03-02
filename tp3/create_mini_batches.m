function [ XB YB ] = create_mini_batches( X, Y, batchSize )
    indices = randperm(size(X,1));
    
    shuffledY = Y(:,indices);
    shuffledX = X(indices,:);
    
    sizes = ones(1, floor(size(X,1)/batchSize)) * batchSize;
    sizes(end) = mod(size(X,1), batchSize) + batchSize;
    
    XB = mat2cell(shuffledX, sizes, size(X,2));
    YB = mat2cell(shuffledY, size(Y,1), sizes);
end