% AGH UST WFiIS
% metody inteligencji obliczeniowej
% temat: 8 - Predykcja zainteresowania postami w social media z użyciem metod NLP 
% grupa: Arkadiusz Trojanowski, Łukasz Kisielewski, Wiktor Gaworek


%%

clc;

Accuracy = 100. - (abs(Difference ./ dataTest(:, idx0 + 6))) * 100.;
mean(Accuracy)

mostPopularWords1 = """" + mostPopularWords + """";

variableNames = ["date" "sentiment" "length" "mentions" "hashtags" "retweets"];
tbl1 = array2table(dataTest(:, 1 : idx0), 'VariableNames', mostPopularWords1);
tbl2 = table(dataTest(:, idx0 + 1), dataTest(:, idx0 + 2), dataTest(:, idx0 + 3), dataTest(:, idx0 + 4), dataTest(:, idx0 + 5), dataTest(:, idx0 + 6), 'VariableNames', variableNames);
tbl = [tbl1 tbl2];
rng('default');
mdl = fitrkernel(tbl, 'retweets', 'CategoricalPredictors', 'all');

queryPoint = tbl(1, :);
explainer = shapley(mdl, tbl, 'QueryPoint', queryPoint, 'UseParallel', true);
explainer.ShapleyValues
plot(explainer);