function [ precision ] = precision( X, Y, Theta )
    Z = repmat(sum(exp(eye(4) * Theta * X)),4,1);
    p = exp(eye(4) * Theta * X) ./ Z;
    [M predictions] = max(p);
    corrects = sum(predictions' == (Y * [1;2;3;4]));
    precision = corrects / size(X,2);
end

