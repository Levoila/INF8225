clc;
clearvars;
load mnist;         %Grayscale data
%load mnist_v2_all; %Binary data

YT = sparse(1:50000,double(train_y)+ones(1,50000),ones(1,50000),50000,10)';
YV = sparse(1:10000,double(valid_y)+ones(1,10000),ones(1,10000),10000,10)';
YTest = sparse(1:10000,double(test_y)+ones(1,10000),ones(1,10000),10000,10)';

%Seems slower with sparse matrices
XT = double(train_x) - 0.5;
XV = double(valid_x) - 0.5;
XTest = double(test_x) - 0.5;


rng(42)
[theta p] = create_and_train_NN(XT, YT, XV, YV, XTest, YTest, [784 800 10], 50, 0.1, 0.0, 0.0, 0.006, 0.0);
