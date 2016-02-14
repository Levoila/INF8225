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
X = [X ; ones(1, 16242)];
X = [X ; randi([0 1], 100, 16242)];
taux_dapprentissage = 0.0005;

[XA XV XT YA YV YT] = create_train_valid_test_splits(X, Y);

batchSize = 568;
alpha = 0.6;

minDiff = 1.0;
minLambdas = [-1 -1];
test = 0;
nb = 1000000;
lambdas = (0.1-0.001).*rand(nb,2)+0.001;
for i = 1:nb
    lambda1 = lambdas(i,1);
    lambda2 = lambdas(i,2);
    
    diff = 0;

    test = test + 1;

    for j = 1:5
        Theta = rand(4,201)-.5;
        nbIterations = 0;
        logVraisemblance = -realmax;
        deltaTheta = zeros(4,201);
        converged = false;
    
        while ~converged
            nbIterations = nbIterations + 1;
            [XB YB] = create_mini_batches(XA, YA, batchSize);

            taux_dapprentissage = 2 / nbIterations;
            for i = 1:size(XB,2)
                %Log vraisemblance
                n = sum(((YV * Theta) .* XV')');
                Z = sum(exp(eye(4) * Theta * XV));
                newLogVraisemblance = sum(n - log(Z));
                delta = newLogVraisemblance - logVraisemblance;
                logVraisemblance = newLogVraisemblance;

                %Gradient
                Z = repmat(sum(exp(eye(4) * Theta * XB{:,i})),4,1);
                p = exp(eye(4) * Theta * XB{:,i}) ./ Z;
                E = p * XB{:,i}';
                goal = YB{i,:}' * XB{:,i}';
                gradient = (E - goal) ./ batchSize + batchSize / size(XA,2) * (lambda1 * 2 * Theta + lambda2 * ((Theta > 0) + (Theta < 0) * -1));
                deltaTheta = alpha*deltaTheta - taux_dapprentissage * gradient;
                Theta = Theta + deltaTheta;

                if abs(delta) < 0.01
                    converged = true;
                    break;
                end
            end
        end

        diff = diff + abs(precision(XA, YA, Theta) - precision(XT, YT, Theta));
    end
    diff = diff / 5;
    fprintf('%d/%d (%f)\n', test, nb, diff);
    
    if diff < minDiff
       minDiff = diff;
       minLambdas = [lambda1 lambda2];
       fprintf('Current min : %f %f (%f)\n', minLambdas(1), minLambdas(2), minDiff);
    end
end


minLambdas 