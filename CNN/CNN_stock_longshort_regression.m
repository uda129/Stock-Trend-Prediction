clear all;
clc;
close all;

%%
SP500_csv= 'datasets\SP500_2000_2017.csv';
DAX_csv = 'datasets\DAX_2000_2017.csv';
NIKKEI_csv = 'datasets\NIKKEI_2000_2017.csv';

window_size=50;
channel=1;

window_num=4000;
train_validation_split=0.7;

%%
SP500_data = readtable(SP500_csv);
DAX_data = readtable(DAX_csv);
NIKKEI_data = readtable(NIKKEI_csv);
REF_LABEL_DURATION=5;
AMPLITUDE_THRESHOULD = 0.05;   %  5%

for window_index= 1:window_num*train_validation_split
    for index_No = 1:4
        trainData(index_No, :, channel, window_index) = table2array(SP500_data(window_index:50-1+window_index, 5));
    end
    
    today = table2array(SP500_data(50-1+window_index, 5));
    tomorrow50 = table2array(SP500_data(50-1+window_index+1:50-1+window_index+1+50-1, 5));

    [trainLabel{window_index, 1}] = find_label(today, tomorrow50, REF_LABEL_DURATION, AMPLITUDE_THRESHOULD) ;
end
trainLabel = categorical(trainLabel);

for window_index= 1+window_num*train_validation_split:window_num
    window_index_from1 = window_index - window_num*train_validation_split;
 
    for index_No = 1:4
        valData(index_No, :, channel, window_index_from1) = table2array(SP500_data(window_index:50-1+window_index, 5));
    end
    
    today = table2array(SP500_data(50-1+window_index, 5));
    tomorrow50 = table2array(SP500_data(50-1+window_index+1:50-1+window_index+1+50-1, 5));
    
    [ valLabel{window_index_from1, 1} ] = find_label(today, tomorrow50, REF_LABEL_DURATION, AMPLITUDE_THRESHOULD) ;
end
valLabel = categorical(valLabel);

%%
layers = [imageInputLayer([4 window_size channel])   % -SP, DAX, NIKKEI  -windows=50, -channel= close 
          convolution2dLayer(2,20)    % -filter size = 2*2, numfilter=20
          reluLayer
          maxPooling2dLayer(2,'Stride',1)   % ~stride=1, padding=False
          fullyConnectedLayer(4)
          softmaxLayer
          classificationLayer()];

%%
MaxEpochs = 30

options = trainingOptions('sgdm','MaxEpochs',MaxEpochs, ...
	'InitialLearnRate',0.0001,...
    'OutputFcn',@plotTrainingAccuracy);

%%
convnet = trainNetwork(trainData, trainLabel,layers,options);

%%
valPredict = classify(convnet,valData);

accuracy = sum(valPredict == valLabel)/numel(valLabel)

%%
valPredictValue = predict(convnet,valData);

%%                                                                                  
SP500_close = table2array(SP500_data(:, 5));
long_count=0;
short_count=0;
probability_threshold=0.7;   %--

[point(1+window_num*train_validation_split+50-1:window_num+50-1 )] = double(valPredict);
point(size(SP500_close))=0;
point = point';

[pointValue(1+window_num*train_validation_split+50-1:window_num+50-1,: )] = double(valPredictValue);
pointValue(size(SP500_close))=0;
pointValue = pointValue';

for num = 1:size(point)
    long_point(num)=nan;
    short_point(num)=nan;
    
    if (point(num) == 1 || point(num) == 3 ) & (pointValue(1, num) > probability_threshold || pointValue(3, num) > probability_threshold)           %--
        long_point(num)= SP500_close(num);
        long_count = long_count +1;
    elseif (point(num) == 2 || point(num) == 4 ) & (pointValue(2, num) > probability_threshold || pointValue(4, num) > probability_threshold)       %--
        short_point(num) = SP500_close(num);
        short_count = short_count +1 ;
    end
end
long_point = long_point';
short_point = short_point';

long_count
short_count

%%
range = [3000:4000];
plot(range,SP500_close(range), range, long_point(range), 'r*', range, short_point(range), 'bla*', range, long_point(range-REF_LABEL_DURATION), 'r+', range, short_point(range-REF_LABEL_DURATION), 'bla+')
plot(range,SP500_close)
%%
function label = find_label(today, tomorrow50, REF_LABEL_DURATION, AMPLITUDE_THRESHOULD)

    future_avg = mean(tomorrow50(1:REF_LABEL_DURATION));
    amplitude = abs((future_avg-today)/today);
    updown = sign(future_avg-today);
    if updown == 0
        label = 'small_long';
    elseif updown == 1 
        if amplitude > AMPLITUDE_THRESHOULD
            label = 'big_long';
        else
            label = 'small_long';
        end
    elseif updown == -1
        if amplitude > AMPLITUDE_THRESHOULD
            label = 'big_short';
        else
            label = 'small_short';
        end
    end

end
%%
function plotTrainingAccuracy(info)

persistent plotObj

if info.State == "start"
    plotObj = animatedline;
    xlabel("Iteration")
    ylabel("Training Accuracy")
elseif info.State == "iteration"
    addpoints(plotObj,info.Iteration,info.TrainingAccuracy)
    drawnow limitrate nocallbacks
end

end

%%
function plotTrainingLoss(info)

persistent plotObj

if info.State == "start"
    plotObj = animatedline;
    xlabel("Iteration")
    ylabel("Training Loss")
elseif info.State == "iteration"
    addpoints(plotObj,info.Iteration,info.TrainingLoss)
    drawnow limitrate nocallbacks
end

end