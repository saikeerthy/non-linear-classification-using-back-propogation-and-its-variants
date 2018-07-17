%% Training Data
clc;
clear;
close all;
N=500;
d=[2 -4 -8];
for i=1:length(d)
train_data1=dbmoon(1000,d(i),10,6); %The function 'dbmoon' produces the data in double moon based on inputs
train_data(:,1)=train_data1(:,1);
train_data(:,2)=train_data1(:,2);
%% Testing Data
test_data1=dbmoon(500,d(i),10,6); %The function 'dbmoon' produces the data in double moon based on inputs
test_data(:,1)=test_data1(:,1);
test_data(:,2)=test_data1(:,2);
target_test_data=test_data1(:,3);
%% Training Multilayer Neural Network
target_data=train_data1(:,3);
net=feedforwardnet(8,'traingd');

if (i==1)
    net.trainParam.lr=0.1;
    rng(10)
elseif (i==2)
    net.trainParam.lr=0.5;
    rng(21)
elseif (i==3)
    net.trainParam.lr=0.7;
    rng(21)
end
net=configure(net,train_data',target_data');

[net,tr]=train(net,train_data',target_data');

y=net(test_data');

figure;
plotconfusion(target_test_data',y);
figure;
plotperform(tr);
%% Decision Boundary
range= -15:0.05:25;
[p1,p2]=meshgrid(range,range);
pp=[p1(:) p2(:)]';
outp = net(pp);
outp(outp<0.5)=0;
outp(outp>=0.5)=1;
figure;
mesh(p1,p2,reshape(outp,length(range),length(range))-5);
x=train_data(:,1);
y=train_data(:,2);
colormap summer;
view(2);
hold on
plot(x(1:1000),y(1:1000),'ob',x(1001:2000),y(1001:2000),'og');
legend('mesh','class 0','class 1');
end