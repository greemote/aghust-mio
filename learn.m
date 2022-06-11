% AGH UST WFiIS
% metody inteligencji obliczeniowej
% temat: 8 - Predykcja zainteresowania postami w social media z użyciem metod NLP 
% grupa: Arkadiusz Trojanowski, Łukasz Kisielewski, Wiktor Gaworek


%%
clc;

% get and fix the prepared data
preppedData1 = preppedData;
idx0 = length(preppedData1(1, :)) - 6;

idx = preppedData1(:, idx0 + 6) == -Inf;
preppedData1(idx, idx0 + 6) = 0;

% create partition
cvp = cvpartition(size(preppedData1,1),'HoldOut',0.2);
dataTrain = preppedData1(training(cvp),:);
dataTest = preppedData1(test(cvp),:);

% set, configure and train the net
net = feedforwardnet;
net.divideFcn = 'dividetrain';

net = configure(net, dataTrain(:, (1 : idx0 + 5))', dataTrain(:, idx0 + 6)');
net = train(net, dataTrain(:, (1 : idx0 + 5))', dataTrain(:, idx0 + 6)');

% get the prediction
YPred = net(dataTest(:, (1 : idx0 + 5))')';

% remove outliers
outliers = isoutlier(YPred(:));
idx = find(outliers == 1);
dataTest(idx, :) = [];
YPred(idx) = [];

% get the difference between test and predicted data
Difference = dataTest(:, idx0 + 6)-YPred;

% plot the results
subplot(3, 1, 1);
plot(dataTest(:, idx0 + 6)); title('test data'); 
subplot(3, 1, 2);
plot(YPred); title('predict'); 
subplot(3, 1, 3);
plot(Difference); title('difference'); 