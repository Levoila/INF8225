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

[XA XV XT] = create_train_valid_test_splits(X);

goal = Y' * X';

converged = false;
for k = 1:25
    %Log vraisemblance
   
    
    %Gradient
    current = zeros(size(groupnames,2), size(X,1));
    %for i = 1:size(X,2)
       %p = exp(eye(4) * Theta * X);
       %d = sum(exp(eye(4) * Theta * X));
       %size(p)
       %size(d)
       %current = current + p1/d * [1 0 0 0]' * X(:,i)' + p2/d * [0 1 0 0]' * X(:,i)' + p3/d * [0 0 1 0]' * X(:,i)' + p4/d * [0 0 0 1]' * X(:,i)';
    %end
    
    %n = exp(Y * Theta * X);
    %d = exp(Y(:,1) * Theta(1,:) * X) + exp(Y(:,2) * Theta(2,:) * X) + exp(Y(:,3) * Theta(3,:) * X) + exp(Y(:,4) * Theta(4,:) * X);
    
    %p = d\n;
    
    %current = p * Y' * X';
    
    
    for i = 1:size(X,2)
       for j = 1:4
           y = [0 0 0 0];
           y(j) = 1;
           n = exp(y * Theta * X(:,i));
           d = 0;
           for k = 1:4
              y2 = [0 0 0 0];
              y2(k) = 1;
              d = d + exp(y2 * Theta * X(:,i));
           end
           
           current = current + n/d * y' * X(:,i)';
       end
    end
    
    gradient = goal - current;
    
    max(max(abs(gradient)))
    
    Theta = Theta + taux_dapprentissage * gradient;
    
    %converged = true;
end

choice = eye(4);

good = 0;
som = [0 0 0 0];
choices = [0 0 0 0];
for i = 1:size(X,2) 
    p1 = exp([1 0 0 0]*Theta*X(:,i));
    p2 = exp([0 1 0 0]*Theta*X(:,i));
    p3 = exp([0 0 1 0]*Theta*X(:,i));
    p4 = exp([0 0 0 1]*Theta*X(:,i));
    
    d = exp([1 0 0 0] * Theta * X(:,i)) + exp([0 1 0 0] * Theta * X(:,i)) + exp([0 0 1 0] * Theta * X(:,i)) + exp([0 0 0 1] * Theta * X(:,i));
    
    p1 = p1/d;
    p2 = p2/d;
    p3 = p3/d;
    p4 = p4/d;
    
    if p1 > p2 && p1 > p3 && p1 > p4
        c = choice(1,:);
    elseif p2 > p1 && p2 > p3 && p2 > p4
        c = choice(2,:);
    elseif p3 > p2 && p3 > p1 && p3 > p4
        c = choice(3,:);
    else
        c = choice(4,:);
    end
    
    choices = choices + c;
    som = som + Y(i,:);
    
    if isequal(c,Y(i,:))
       good = good + 1; 
    end
end

good / size(X,2)
som
choices