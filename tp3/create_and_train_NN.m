function [ THETA ] = create_and_train_NN( XT, YT, XV, YV, hiddenLayers )
    batchSize = 1000;
    converged = false;
    learningRate = 0.001;
    
    for i = 1:hiddenLayers
       THETA{i} = rand(size(XT,2),size(XT,2)) - 0.5; 
    end
    THETA{i+1}  = rand(10,size(XT,2)) - 0.5;
    
    while ~converged
       [XB, YB] = create_mini_batches(XT, YT, batchSize);
       for i = 1:size(XB,1)
           %Forward propagation
           h{1} = XB{i}';
           for j = 2:hiddenLayers+1
               a{j-1} = THETA{j-1} * h{j-1};
               h{j} = (a{j-1} >= 0) .* a{j-1} + 0.1 * (a{j-1} < 0) .* a{j-1};
           end
           a{j} = THETA{j} * h{j};
           
           f = exp(a{j}) ./ repmat(sum(exp(a{j}),1),size(a{j},1),1);
           delta{j} = -(YB{i} - f);

           %backward propagation
           for j = hiddenLayers:-1:1
               D = (a{j} >= 0) + 0.1 * (a{j} < 0);
               delta{j} = D .* (THETA{j+1}' * delta{j+1});
           end

           for j = 1:hiddenLayers+1
               gradient = -delta{j} * h{j}' ./ batchSize;
               THETA{j} = THETA{j} + learningRate * gradient;
           end
           fprintf('%d\n', i);
           
           
           %Check precision over validation set
           h{1} = XV';
           for j = 2:hiddenLayers+1
               a{j-1} = THETA{j-1} * h{j-1};
               h{j} = (a{j-1} >= 0) .* a{j-1} + 0.1 * (a{j-1} < 0) .* a{j-1};
           end
           a{j} = THETA{j} * h{j};

           f = exp(a{j}) ./ repmat(sum(exp(a{j}),1),size(a{j},1),1);
           [M predictions] = max(f);
           precision = sum(sum(predictions' == (YV' * [1;2;3;4;5;6;7;8;9;10]))) / size(YV,2)
       end
    end
end

