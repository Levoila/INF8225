clc;
clearvars;
load mnist_all;

YT = sparse(1:50000,double(train_y)+ones(1,50000),ones(1,50000),50000,10)';
YV = sparse(1:10000,double(valid_y)+ones(1,10000),ones(1,10000),10000,10)';

create_and_train_NN(im2double(train_x), YT, im2double(valid_x), YV, 2);