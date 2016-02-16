clc;
load 20news_w100;
n = 4;
m = size(newsgroups,2);
o = ones(1,m);
i = 1:m;
j = newsgroups;
Y = sparse(i,j,o,m,n);

Theta = rand(4,101)-.5;
X = documents;
X = [X ; ones(1, 16242)];
taux_dapprentissage = 0.0005;

[XA XV XT YA YV YT] = create_train_valid_test_splits(X, Y);


%Approche par batch
goal = YA' * XA';

nbIterations = 0;
logVraisemblance = -realmax;
precisionApprentissage = [];
precisionValidation = [];
lvBatch = [];
while true
    %Log vraisemblance
    n = sum(((YV * Theta) .* XV')');
    Z = sum(exp(Theta * XV));
    newLogVraisemblance = sum(n - log(Z));
    delta = newLogVraisemblance - logVraisemblance;
    logVraisemblance = newLogVraisemblance;
    lvBatch = [lvBatch logVraisemblance];
    
    precisionApprentissage = [precisionApprentissage precision(XA, YA, Theta)];
    precisionValidation = [precisionValidation precision(XV, YV, Theta)];
    
    %Gradient
    Z = repmat(sum(exp(Theta * XA)),4,1);
    p = exp(Theta * XA) ./ Z;
    E = p * XA';
    gradient = E - goal;
    
    %Mise � jour des param�tres
    Theta = Theta - taux_dapprentissage * gradient;
    
    nbIterations = nbIterations + 1;
    if abs(delta) < 1
        break;
    end
end
x1 = 1:nbIterations;

precisionTest = precision(XT, YT, Theta);

fprintf('Pr�cision sur l''ensemble de tests pour la descente par batch = %f\n', precisionTest);

%Approche par mini-batches
Theta = rand(4,101)-.5;
batchSize = 568;
alpha = 0.6;
deltaTheta = zeros(4,101);
nbIterations = 0;
logVraisemblance = -realmax;
converged = false;
lvMiniBatches = [];
pvMiniBatches = [];
paMiniBatches = [];
while ~converged
    nbIterations = nbIterations + 1;
    [XB YB] = create_mini_batches(XA, YA, batchSize);
    
    taux_dapprentissage = 2 / nbIterations;
    for i = 1:size(XB,2)
        %Log vraisemblance
        n = sum(((YV * Theta) .* XV')');
        Z = sum(exp(Theta * XV));
        newLogVraisemblance = sum(n - log(Z));
        delta = newLogVraisemblance - logVraisemblance;
        logVraisemblance = newLogVraisemblance;
        lvMiniBatches = [lvMiniBatches logVraisemblance];
        
        %Gradient
        Z = repmat(sum(exp(Theta * XB{:,i})),4,1);
        p = exp(Theta * XB{:,i}) ./ Z;
        E = p * XB{:,i}';
        goal = YB{i,:}' * XB{:,i}';
        gradient = (E - goal) ./ batchSize;
        
        %Mise � jour des param�tres
        deltaTheta = alpha*deltaTheta - taux_dapprentissage * gradient;
        Theta = Theta + deltaTheta;
        
        pvMiniBatches = [pvMiniBatches precision(XV, YV, Theta)];
        
        if abs(delta) < 0.0001
            converged = true;
            break;
        end
    end
    paMiniBatches = [paMiniBatches precision(XA, YA, Theta)];
end
x2 = (1:size(lvMiniBatches,2)) / size(XB,2);
x3 = 1:nbIterations;

precisionTest = precision(XT, YT, Theta);

fprintf('Pr�cision sur l''ensemble de tests pour les mini-batches = %f\n', precisionTest);

figure();
plot(x1,lvBatch,x2,lvMiniBatches);
title('Log vraisemblance en fonction du num�ro d''it�ration');
xlabel('Num�ro d''it�ration');
ylabel('Log vraisemblance');
legend('Batch','Mini Batches');

figure();
plot(x1,precisionApprentissage,x1,precisionValidation,x2,pvMiniBatches,x3,paMiniBatches);
title('Courbes d''apprentissage');
xlabel('Num�ro d''it�ration');
ylabel('Pr�cision');
legend('Ensemble d''apprentissage (Batch)','Ensemble de validation (Batch)', 'Ensemble de validation (Mini Batches)', 'Ensemble d''apprentissage (Mini Batches)');