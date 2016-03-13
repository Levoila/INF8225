function [ bestTheta, precision ] = create_and_train_NN( XT, YT, XV, YV, XTest, YTest, layerSizes, batchSize, learningRate, lambda1, lambda2, pwA, dropoutRate )
    converged = false;
    hiddenLayers = size(layerSizes,2)-2;
    
    %Random weights initialization
    for i = 2:size(layerSizes,2)
       THETA{i-1} = (rand(layerSizes(i),layerSizes(i-1)+1) - 0.5) / sqrt(layerSizes(i-1)+1);
    end
    
    figure();
    title('Précision du réseau de neurones');
    lHandle1 = line(NaN, NaN, 'Color', 'Blue');
    lHandle2 = line(NaN, NaN, 'Color', 'Red');
    legend('Training','Validation', 'Location', 'southeast');
    xlabel('Epoch')
    ylabel('Précision')
    axis([0 inf 0.9 1.0]);
    pause(0.01);
    
    t = 0;
    maxP = 0;
    minLoss = 100.0;
    pT = [];
    pV = [];
    loss = [];
    timeToLive = 10; %Nombre d'itération une fois que la précision sur l'ensemble d'apprentissage dépasse 0.999
    while ~converged
        t = t + 1;
        
       [XB, YB] = create_mini_batches(XT, YT, batchSize);
       for i = 1:size(XB,1)
           %Forward propagation
           [a, h, f, averageLoss] = forward_propagation(XB{i}, YB{i}, THETA, pwA, true, dropoutRate);
           
           delta{hiddenLayers+1} = -(YB{i} - f);

           %backward propagation
           for j = hiddenLayers:-1:1
               D = (a{j} >= 0) + pwA * (a{j} < 0);
               delta{j} = D .* (THETA{j+1}(:,1:end-1)' * delta{j+1});
           end

           for j = 1:hiddenLayers+1
               %Parameters update using L1 and L2 regularization
               THETA{j} = (1 - learningRate * lambda2 * batchSize / size(XT,1)) * THETA{j} - (learningRate / batchSize) * delta{j} * h{j}' - learningRate * lambda1 * batchSize / size(XT,1) * sign(THETA{j});
           end
       end
       
       %Check precision over validation set
       [precision, averageLoss] = compute_precision(XV, YV, THETA, pwA, dropoutRate);
       loss = [loss averageLoss];
       pV = [pV precision];
       if precision > maxP
           maxP = precision;
           bestTheta = THETA;
           fprintf('Max : %f\n', maxP);
       end
       
       if averageLoss < minLoss
           minLoss = averageLoss;
       end
       
       %Check precision over training set
       [precision averageLoss] = compute_precision(XT, YT, THETA, pwA, dropoutRate);
       pT = [pT precision];
       fprintf('Training : %f (%d)\n', precision, t);
       
       fprintf('Validation : %f( max : %f ) \tLoss : %f( min : %f ) (%d)\n', precision, maxP, full(averageLoss), full(minLoss), t);
        
       %Mise-à-jour en temps réel du graphique
       set(lHandle1, 'XData', 1:t, 'YData', pT);
       set(lHandle2, 'XData', 1:t, 'YData', pV);
       pause(0.01);
       
       if precision >= 0.999
           timeToLive = timeToLive - 1;
           if timeToLive == 0
               converged = true
           end
       end
    end
    
    [precision averageLoss] = compute_precision(XTest, YTest, bestTheta, pwA, dropoutRate);
    fprintf('Précision sur l''ensemble de test : %f\n', precision, precision2);
end