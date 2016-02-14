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
lv = [];
% while true
%     %Log vraisemblance
%     n = sum(((YA * Theta) .* XA')');
%     Z = sum(exp(eye(4) * Theta * XA));
%     newLogVraisemblance = sum(n - log(Z));
%     delta = newLogVraisemblance - logVraisemblance;
%     logVraisemblance = newLogVraisemblance;
%     lv = [lv logVraisemblance];
%     
%     precisionApprentissage = [precisionApprentissage precision(XA, YA, Theta)];
%     precisionValidation = [precisionValidation precision(XV, YV, Theta)];
%     
%     %Gradient
%     Z = repmat(sum(exp(eye(4) * Theta * XA)),4,1);
%     p = exp(eye(4) * Theta * XA) ./ Z;
%     E = p * XA';
%     gradient = E - goal;
%     Theta = Theta - taux_dapprentissage * gradient;
%     
%     nbIterations = nbIterations + 1;
%     if delta < 5
%         break;
%     end
% end

precisionTest = precision(XT, YT, Theta);

fprintf('Précision sur l''ensemble de tests = %f\n', precisionTest);


%Approche par mini-batch
Theta = rand(4,101)-.5;
batchSize = 568;

nbIterations = 1;
while true
    
    [XB YB] = create_mini_batches(XA, YA, batchSize);
    
    for i = 1:size(XB,2)
        %Log vraisemblance
        n = sum(((YB{i,:} * Theta) .* XB{:,i}')');
        Z = sum(exp(eye(4) * Theta * XB{:,i}));
        newLogVraisemblance = sum(n - log(Z));
        delta = newLogVraisemblance - logVraisemblance;
        logVraisemblance = newLogVraisemblance;
        
        %Gradient
        Z = repmat(sum(exp(eye(4) * Theta * XB{:,i})),4,1);
        p = exp(eye(4) * Theta * XB{:,i}) ./ Z;
        E = p * XB{:,i}';
        goal = YB{i,:}' * XB{:,i}';
        gradient = (E - goal) ./ batchSize;
        Theta = Theta - taux_dapprentissage * gradient;
        
        precisionValidation = precision(XV, YV, Theta);
    end
    precisionValidation
end

nbIterations
x = 1:nbIterations;

figure();
plot(x,lv);
title('Log vraisemblance en fonction du numéro d''itération');
xlabel('Numéro d''itération');
ylabel('Log vraisemblance');

figure();
plot(x,precisionApprentissage,x,precisionValidation);
title('Courbes d''apprentissage');
xlabel('Numéro d''itération');
ylabel('Précision');
legend('Ensemble d''apprentissage','Ensemble de validation');