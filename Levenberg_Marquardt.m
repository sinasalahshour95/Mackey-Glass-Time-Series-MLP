clc;close all;clear;
%% 
tic;
data = readmatrix('normalize_dataset.csv');
num_data = size(data,1);
data = [ones(num_data,1) data];  %data with bias input

num_input = size(data,2)-1;

percent_train = 0.65;
percent_validation = 0.15;
percent_test = 0.25;

num_train = round(num_data*percent_train);
num_validation = round(num_data*percent_validation);
num_test = num_data-(num_train+num_validation);

n1 = num_input; %number of input
n2 = 5;         %number of neurons in the hidden layer
n3 = 1;         %number of neurons in the output layer


epoch = 40;
mse_train = zeros(1,epoch);
mse_test = zeros(1,epoch);
mse_validation = zeros(1,epoch);

w1 = readmatrix('initial_w1.csv');
jacob_w1 = zeros(num_train,n1 * n2);

w2 = readmatrix('initial_w2.csv');
jacob_w2 = zeros(num_train,n2);

num_weights = (n1 * n2) + (n2 * n3);

I = eye(num_weights);
jacob = zeros(num_train,num_weights,1);

train_inputs = data(1:num_train, 1:num_input)';
train_targets = data(1:num_train, end)';

validation_inputs = data(num_train+1:num_train+num_validation, 1:num_input)';
validation_targets = data(num_train+1:num_train+num_validation, end)';

test_inputs = data(num_train+num_validation+1:end, 1:num_input)';
test_targets = data(num_train+num_validation+1:end, end)';

miu_initial = 1;
eta = 1;

for t = 1:epoch
    
    net1 = w1*train_inputs;
    o1 = logsig(net1);
    net2 = w2*o1;
    o2 = net2; % Linear output layer
    error = train_targets-o2;
    
    d_logsig1 = o1.*(1-o1); %derivative of first active function
 
    jacob_w2 = -o1'; %-1*1*o1'
        
    for i=1:num_train
        
        w2_d_logsig1 = w2 .* d_logsig1(:,i)';
        dw1 = w2_d_logsig1' * train_inputs(:,i)';
        jacob_w1(i,:) = -reshape(dw1, 1, n1 * n2); %-1*(w2*A)'*input
    end
    
    jacob = [jacob_w1, jacob_w2];
    
    w=[w1(:);w2(:)]';
        
    miu = miu_initial * sum(error.^2);
    jtj = jacob' * jacob;
    je = jacob' * error';
    dw = (jtj + miu * I) \ je;
    
    w = w - eta * dw';
    
    w1 = reshape(w(1:n2*n1),n2,n1);
    w2 = reshape(w(n2*n1+1:end),n3,n2);
    
    %MSE Train Part
    output_data_train = w2 * logsig(w1 * train_inputs); 
    mse_train(t) = mse(train_targets - output_data_train);
    
    %MSE Validation Part
    
    output_data_validation = w2 * logsig(w1 * validation_inputs); 
    mse_validation(t)= mse(validation_targets - output_data_validation);
    
    %MSE Test Part
    output_data_test = w2 * logsig(w1 * test_inputs); 
    mse_test(t) = mse(test_targets - output_data_test);
    
    % Plot MSE Curves
    figure(1);

    semilogy(1:epoch, mse_train, 'b', 'LineWidth', 2);
    hold on;
    semilogy(1:epoch, mse_validation, 'g', 'LineWidth', 2);
    semilogy(1:epoch, mse_test, 'r', 'LineWidth', 2);
    hold off;

    title('Mean Squared Error vs. Epoch');
    xlabel('Epoch');
    ylabel('MSE (log scale)');
    legend('Training', 'Validation', 'Test');
    grid on;

    % Plot Target vs. Output
    figure(2);

    subplot(3,1,1);
    plot(train_targets, 'b-', 'LineWidth', 1); hold on; plot(output_data_train, 'r--', 'LineWidth', 1); hold off;
    legend('Target', 'Output'); 
    title('Training Data');

    subplot(3,1,2);
    plot(validation_targets, 'g-', 'LineWidth', 1); hold on; plot(output_data_validation, 'r--', 'LineWidth', 1); hold off;
    legend('Target', 'Output'); 
    title('Validation Data');

    subplot(3,1,3);
    plot(test_targets, 'k-', 'LineWidth', 1); hold on; plot(output_data_test, 'r--', 'LineWidth', 1); hold off;
    legend('Target', 'Output'); 
    title('Test Data');
    
    pause(0.1);
end

% Plot Regression Plots
figure('Name', 'Regression Analysis Train');
plotregression(train_targets, output_data_train); title('Training');
figure('Name', 'Regression Analysis Validation');
plotregression(validation_targets, output_data_validation); title('Validation');
figure('Name', 'Regression Analysis Test');
plotregression(test_targets, output_data_test); title('Test');
   
toc;

