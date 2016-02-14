function [ XB YB ] = create_mini_batches( X, Y, batchSize )
    indices = randperm(size(X,2));
    
    shuffledY = Y(indices,:);
    shuffledX = X(:,indices);
    
    sizes = ones(1, floor(size(X,2)/batchSize)) * batchSize;
    sizes(end) = mod(size(X,2), batchSize) + batchSize;
    
    XB = mat2cell(shuffledX, size(X,1), sizes);
    YB = mat2cell(shuffledY, sizes, size(Y,2));
end

