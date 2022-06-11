% AGH UST WFiIS
% metody inteligencji obliczeniowej
% temat: 8 - Predykcja zainteresowania postami w social media z użyciem metod NLP 
% grupa: Arkadiusz Trojanowski, Łukasz Kisielewski, Wiktor Gaworek


%%
clear;clc;

% reading data
trumpTweets = readtable('realdonaldtrump.csv');

sampleSize = length(trumpTweets.id(:));
trumpTweets = trumpTweets(1 : sampleSize, 3:8);

% extracting sentiments of tweets
% training sentiment analysis net
emb = fastTextWordEmbedding;
data = readLexicon;

idx = ~isVocabularyWord(emb,data.Word);
data(idx,:) = [];

idx = data{:, 2} == "Positive";
positiveKnown = table2array(data(idx,1));
idx = data{:, 2} == "Negative";
negativeKnown = table2array(data(idx,1));

dataTrain = data;
wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

mdl = fitcknn(XTrain,YTrain);

% analysing sentiments of tweets
% predicting sentiments of words in tweets
documents = preprocessText(trumpTweets.content);

enc = wordEncoding(documents);

idx = ~isVocabularyWord(emb, documents.Vocabulary);
documents = removeWords(documents,idx);

words = documents.Vocabulary;
words(ismember(words, wordsTrain)) = [];

%%
% creating the set of vectors of the most impactful words
[~, mostPopularIdx] = maxk(trumpTweets.retweets, 5);
mostPopularDocuments = documents(mostPopularIdx);
mostPopularWords = mostPopularDocuments.Vocabulary;

tweetContent = zeros(sampleSize, length(mostPopularWords));
for i = 1 : sampleSize
    for j = 1 : length(mostPopularWords)
        if ismember(mostPopularWords(j), documents(i).string)
            tweetContent(i, j) = 1;
        end
    end
end

vec = word2vec(emb,words);
SentimentPrediction = predict(mdl,vec);

%%
% assigning sentiment values to tweets
positivePredicted = words(SentimentPrediction == "Positive")';
negativePredicted = words(SentimentPrediction == "Negative")';
positiveAll = [positiveKnown; positivePredicted];
negativeAll = [negativeKnown; negativePredicted];

sentiments = zeros(sampleSize, 1);
tweetLengths = zeros(sampleSize, 1);
for idx = 1 : sampleSize
    tweetWords = documents(idx).Vocabulary';
    tweetLengths(idx) = tweetWords.size(1);
    for w = 1 : tweetWords.size()
        if ismember(tweetWords(w), positiveAll)
            sentiments(idx) = sentiments(idx) + 1;
        elseif ismember(tweetWords(w), negativeAll)
            sentiments(idx) = sentiments(idx) - 2;
        end
    end
end

%% 
% counting the number of mentions and hashtags in every tweet
mentionsCount = zeros(sampleSize, 1);
hashtagsCount = zeros(sampleSize, 1);
for i = 1 : sampleSize
    mentionsCount(i) = length(strfind(string(trumpTweets.mentions(i)), '@'));
    hashtagsCount(i) = length(strfind(string(trumpTweets.hashtags(i)), '#'));
end

% dividing the tweets by the date
presidentialIndex = datenum(datetime(trumpTweets.date(:), 'InputFormat', 'yyyy-MM-dd')) >= datenum(datetime('16.06.2015'));

% merging gathered data
preppedData = [tweetContent presidentialIndex abs(sentiments) tweetLengths mentionsCount hashtagsCount log(trumpTweets.retweets)];


%%
function data = readLexicon

% read positive words
fidPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt'));
C = textscan(fidPositive,'%s','CommentStyle',';');
wordsPositive = string(C{1});

% read negative words
fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
C = textscan(fidNegative,'%s','CommentStyle',';');
wordsNegative = string(C{1});
fclose all;

% create table of labeled words
words = [wordsPositive;wordsNegative];
labels = categorical(nan(numel(words),1));
labels(1:numel(wordsPositive)) = "Positive";
labels(numel(wordsPositive)+1:end) = "Negative";

data = table(words,labels,'VariableNames',{'Word','Label'});

end

function documents = preprocessText(textData)

% tokenize the text
documents = tokenizedDocument(textData, 'Language', 'en');

% erase punctuation
documents = erasePunctuation(documents);

% remove a list of stop words
documents = removeStopWords(documents);

% convert to lowercase
documents = lower(documents);

end