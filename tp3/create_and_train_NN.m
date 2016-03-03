function [ THETA ] = create_and_train_NN( XT, YT, XV, YV, XTest, YTest, layerSizes, batchSize, learningRate, lambda1, lambda2 )
    converged = false;
    %learningRate = 0.1;
    hiddenLayers = size(layerSizes,2)-2;
    
    for i = 2:size(layerSizes,2)
       THETA{i-1} = (rand(layerSizes(i),layerSizes(i-1)+1) - 0.5) / sqrt(layerSizes(i-1)+1);
    end
    
    t = 0;
    maxP = 0;
    pT = [];
    pV = [];
    n = 1000;
    %while ~converged
    for k = 1:n
        t = t + 1;
       [XB, YB] = create_mini_batches(XT, YT, batchSize);
       for i = 1:size(XB,1)
           %Forward propagation
           [a, h, f, averageLoss] = forward_propagation(XB{i}, YB{i}, THETA);
           
           delta{hiddenLayers+1} = -(YB{i} - f);

           %backward propagation
           for j = hiddenLayers:-1:1
               D = (a{j} >= 0) + 0.1 * (a{j} < 0);
               delta{j} = D .* (THETA{j+1}(:,1:end-1)' * delta{j+1});
           end

           for j = 1:hiddenLayers+1
               %Parameters update using L1 and L2 regularization
               THETA{j} = (1 - learningRate * lambda2 * batchSize / size(XT,1)) * THETA{j} - (learningRate / batchSize) * delta{j} * h{j}' - learningRate * lambda1 * batchSize / size(XT,1) * sign(THETA{j});
           end
       end
       
       %Check precision over validation set
       precision = compute_precision(XV, YV, THETA);
       pV = [pV precision];
       if precision > maxP
           maxP = precision;
           bestTheta = THETA;
           fprintf('Max : %f (%d)\n', maxP, t);
       end
       fprintf('Validation : %f (%d)\n', precision, t);
       
       %Check precision over validation set
       precision = compute_precision(XT, YT, THETA);
       pT = [pT precision];
       fprintf('Training : %f (%d)\n', precision, t);
    end
    
    precision = compute_precision(XTest, YTest, bestTheta);
    fprintf('Test : %f\n', precision);
    
    figure();
    t = sprintf('Precision with learning rate = %f', learningRate);
    plot(1:n,pV,1:n,pT);
    legend('Validation', 'Test');
    title(t);
end