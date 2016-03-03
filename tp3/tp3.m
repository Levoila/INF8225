clc;
clearvars;
load mnist_all_v2;

YT = sparse(1:50000,double(train_y)+ones(1,50000),ones(1,50000),50000,10)';
YV = sparse(1:10000,double(valid_y)+ones(1,10000),ones(1,10000),10000,10)';
YTest = sparse(1:10000,double(test_y)+ones(1,10000),ones(1,10000),10000,10)';

%Seems slower with sparse matrices
XT = logical(train_x);
XV = logical(valid_x);
XTest = logical(test_x);

%create_and_train_NN(im2double(train_x), YT, im2double(valid_x), YV, [784 150 10], 1000, 0.5, 50.0);
create_and_train_NN(XT, YT, XV, YV, XTest, YTest, [784 500 10], 50, 0.001, 0.0, 1.5);
