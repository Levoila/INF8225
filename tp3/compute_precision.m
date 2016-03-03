function [ precision ] = compute_precision( X, Y, THETA )
    [a, h, f, averageLoss] = forward_propagation(X, Y, THETA);
    [M predictions] = max(f);
    precision = sum(sum(predictions' == (Y' * [1;2;3;4;5;6;7;8;9;10]))) / size(Y,2);
end

