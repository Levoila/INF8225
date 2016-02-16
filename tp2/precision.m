function [ precision ] = precision( X, Y, Theta )
    Z = repmat(sum(exp(Theta * X)),4,1);
    p = exp(Theta * X) ./ Z;
    [M predictions] = max(p);
    corrects = sum(predictions' == (Y * [1;2;3;4]));
    precision = corrects / size(X,2);
end

