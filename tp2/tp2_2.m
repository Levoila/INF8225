clc;
load 20news_w100;
n = 4;
m = size(newsgroups,2);
o = ones(1,m);
i = 1:m;
j = newsgroups;
Y = sparse(i,j,o,m,n);

Theta = rand(4,201)-.5;
X = documents;
X = [X ; randi([0 1], 100, 16242)];
X = [X ; ones(1, 16242)];

[XA XV XT YA YV YT] = create_train_valid_test_splits(X, Y);

%Mini-batch without regularization
batchSize = 568;
alpha = 0.6;
deltaTheta = zeros(4,201);
nbIterations = 0;
logVraisemblance = -realmax;
converged = false;
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
        
        %Gradient
        Z = repmat(sum(exp(Theta * XB{:,i})),4,1);
        p = exp(Theta * XB{:,i}) ./ Z;
        E = p * XB{:,i}';
        goal = YB{i,:}' * XB{:,i}';
        gradient = (E - goal) ./ batchSize;
        
        %Mise � jour des param�tres
        deltaTheta = alpha*deltaTheta - taux_dapprentissage * gradient;
        Theta = Theta + deltaTheta;
        
        if abs(delta) < 0.01
            converged = true;
            break;
        end
    end
end

precisionTest = precision(XT, YT, Theta);
fprintf('Pr�cision sur l''ensemble de test sans r�gularisation = %f\n', precisionTest);

figure();
subplot(2,2,1);
histogram(abs(Theta(1,1:101)), 20);
title({'Histogramme des param�tres originaux (Y = 1)';'sans terme de r�gularisation'});
xlabel('Poid');
ylabel('Occurences');
subplot(2,2,2);
histogram(abs(Theta(2,1:101)), 20);
title({'Histogramme des param�tres originaux (Y = 2)';'sans terme de r�gularisation'});
xlabel('Poid');
ylabel('Occurences');
subplot(2,2,3);
histogram(abs(Theta(3,1:101)), 20);
title({'Histogramme des param�tres originaux (Y = 3)';'sans terme de r�gularisation'});
xlabel('Poid');
ylabel('Occurences');
subplot(2,2,4);
histogram(abs(Theta(4,1:101)), 20);
title({'Histogramme des param�tres originaux (Y = 4)';'sans terme de r�gularisation'});
xlabel('Poid');
ylabel('Occurences');

figure();
histogram(abs(Theta(:,102:end)), 20);
title({'Histogramme des param�tres al�atoires';'sans terme de r�gularisation'});
xlabel('Poid');
ylabel('Occurences');

%Mini-batch with regularization
Theta = rand(4,201)-.5;
batchSize = 568;
alpha = 0.6;
deltaTheta = zeros(4,201);
lambda2 = 0.016;
lambda1 = 0.046;
nbIterations = 0;
logVraisemblance = -realmax;
converged = false;

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
        
        %Gradient
        Z = repmat(sum(exp(Theta * XB{:,i})),4,1);
        p = exp(Theta * XB{:,i}) ./ Z;
        E = p * XB{:,i}';
        goal = YB{i,:}' * XB{:,i}';
        gradient = (E - goal) ./ batchSize + batchSize / size(XA,2) * (lambda2 * 2 * Theta + lambda1 * ((Theta > 0) + (Theta < 0) * -1));
        
        %Mise � jour des param�tres
        deltaTheta = alpha*deltaTheta - taux_dapprentissage * gradient;
        Theta = Theta + deltaTheta;
        
        if abs(delta) < 0.01
            converged = true;
            break;
        end
    end
end

precisionTest = precision(XT, YT, Theta);
fprintf('Pr�cision sur l''ensemble de test avec r�gularisation = %f\n', precisionTest);

figure();
subplot(2,2,1);
histogram(abs(Theta(1,1:101)), 20);
title({'Histogramme des param�tres originaux (Y = 1)';'avec terme de r�gularisation'});
xlabel('Poid');
ylabel('Occurences');
subplot(2,2,2);
histogram(abs(Theta(2,1:101)), 20);
title({'Histogramme des param�tres originaux (Y = 2)';'avec terme de r�gularisation'});
xlabel('Poid');
ylabel('Occurences');
subplot(2,2,3);
histogram(abs(Theta(3,1:101)), 20);
title({'Histogramme des param�tres originaux (Y = 3)';'avec terme de r�gularisation'});
xlabel('Poid');
ylabel('Occurences');
subplot(2,2,4);
histogram(abs(Theta(4,1:101)), 20);
title({'Histogramme des param�tres originaux (Y = 4)';'avec terme de r�gularisation'});
xlabel('Poid');
ylabel('Occurences');

figure();
histogram(abs(Theta(:,102:end)), 20);
title({'Histogramme des param�tres al�atoires';'avec terme de r�gularisation'});
xlabel('Poid');
ylabel('Occurences');








