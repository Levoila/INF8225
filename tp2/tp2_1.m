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

goal = YA' * XA';

nbIterations = 0;
logVraisemblance = -realmax;
precisionApprentissage = [];
precisionValidation = [];
lv = [];
while true
    %Log vraisemblance
    n = sum(((Y * Theta) .* X')');
    Z = sum(exp(eye(4) * Theta * X));
    newLogVraisemblance = sum(n - log(Z));
    delta = newLogVraisemblance - logVraisemblance;
    logVraisemblance = newLogVraisemblance;
    lv = [lv logVraisemblance];
    
    precisionApprentissage = [precisionApprentissage precision(XA, YA, Theta)];
    precisionValidation = [precisionValidation precision(XV, YV, Theta)];
    
    %Gradient
    Z = repmat(sum(exp(eye(4) * Theta * XA)),4,1);
    p = exp(eye(4) * Theta * XA) ./ Z;
    E = p * XA';
    gradient = E - goal;
    Theta = Theta - taux_dapprentissage * gradient;
    
    nbIterations = nbIterations + 1;
    if delta < 0.1
        break;
    end
end

precisionTest = precision(XT, YT, Theta);

nbIterations
x = 1:nbIterations;

figure();
loglog(x,lv);
title('Log vraisemblance en fonction du numéro d''itération');
xlabel('Numéro d''itération');
ylabel('Log vraisemblance');

figure();
loglog(x,precisionApprentissage,x,precisionValidation);
title('Courbes d''apprentissage');
xlabel('Numéro d''itération');
ylabel('Précision');
legend('Ensemble d''apprentissage','Ensemble de validation');