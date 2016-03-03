function [ a, h, f, averageLoss ] = forward_propagation( X, Y, THETA )
    h{1} = [X' ; ones(1,size(X,1))];
   for j = 2:size(THETA,2)
       a{j-1} = THETA{j-1} * h{j-1};
       h{j} = [(a{j-1} >= 0) .* a{j-1} + 0.1 * (a{j-1} < 0) .* a{j-1} ; ones(1,size(a{j-1},2))];
   end
   a{j} = THETA{j} * h{j};
   
   %Numerically stable version of softmax
   %From http://stackoverflow.com/questions/9906136/implementation-of-a-softmax-activation-function-for-neural-networks
   % and http://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
   maxValue = repmat(max(a{j},[],1),size(a{j},1),1);
   f = exp(a{j} - maxValue - repmat(log(sum(exp(a{j} - maxValue))),size(a{j},1),1));

   averageLoss = -sum(sum(Y .* log(f))) / size(X,1);
end

