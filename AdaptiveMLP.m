clc;close all;clear;
%% 
data = readmatrix('normalize_dataset.csv');
num_data = size(data,1);
data = [ones(num_data,1) data];  % data with bias input

num_input = size(data,2)-1;

percent_train = 0.65;
percent_validation = 0.15;
percent_test = 0.25;

num_train = round(num_data*percent_train);
num_validation = round(num_data*percent_validation);
num_test = num_data-(num_train+num_validation);

n1 = num_input; % number of input
n2 = 5;         % number of neurons in the hidden layer
n3 = 1;         % number of neurons in the output layer

landa = 0.001;     % learning rate of eta

epoch = 700;
mse_train = zeros(1,epoch);
mse_test = zeros(1,epoch);
mse_validation = zeros(1,epoch);

w1 = readmatrix('initial_w1.csv'); % initial W1
eta1 = repmat(0.05,[n2 n1]);       % learning rate for w1
net1 = zeros(n2,1);
o1 = zeros(n2,1);

w2 = readmatrix('initial_w2.csv'); % initial W2
eta2 = repmat(0.05,[n3 n2]);       % learning rate for w2
net2 = zeros(n3,1);
o2 = zeros(n3,1);

train_inputs = data(1:num_train, 1:num_input);
train_targets = data(1:num_train, end);

validation_inputs = data(num_train+1:num_train+num_validation, 1:num_input);
validation_targets = data(num_train+1:num_train+num_validation, end);

test_inputs = data(num_train+num_validation+1:end, 1:num_input);
test_targets = data(num_train+num_validation+1:end, end);

for t = 1:epoch  
    
        %Training Part
        error = zeros(1,num_train);
        for i = 1:num_train
            
            input = train_inputs(i,:);
            net1 = w1*input';
            o1 = logsig(net1);
            
            net2 = w2*o1;
            o2 = net2;
            
            target = train_targets(i,1);
            error(i) = target-o2;

            t1 = o1.*(1-o1); %derivative of first active function
            A = diag(t1);    %Matrix
            
            if i ~= 1
               eta2 = eta2 + landa*((error(i)*1*-1*o1') .* (error(i-1)*1*-1*o1_p')); 
               eta1 = eta1 + landa*((error(i)*-1*(w2*A)'*input) .* (error(i-1)*-1*(w2_p*A_p)'*input_p));
            end
            
            input_p = input; %previous Input
            o1_p = o1;       %previous output of hidden layer
            o2_p = o2;       %previous output of output layer
            A_p = A;         %previous derivative of first active function
            w2_p = w2;       %previous w2
            
            w1 = w1 + eta1 .* (error(i) * (w2 * A)' * input);  % w1 - eta1 .* (error(i) * -1 * (w2 * A)' * input)
            w2 = w2 + eta2 .* (error(i) * o1');                % w2 - eta2 .* (error(i) * -1 * 1 * o1')      
   
        end
        
        %MSE Train Part
        output_data_train = w2 * logsig(w1 * train_inputs');
        error_data_train = train_targets' - output_data_train;
        mse_train(t) = mse(error_data_train);
        
        %MSE Validation Part 
        output_data_validation = w2 * logsig(w1 * validation_inputs');
        error_data_validation = validation_targets' - output_data_validation;
        mse_validation(t) = mse(error_data_validation);
         
        %MSE Test Part
        output_data_test = w2 * logsig(w1 * test_inputs');
        error_data_test = test_targets' - output_data_test;
        mse_test(t) = mse(error_data_test);
              
        % Plot MSE Curves
        figure(1);

        semilogy(1:t, mse_train(1:t), 'b', 'LineWidth', 2);
        hold on;
        semilogy(1:t, mse_validation(1:t), 'g', 'LineWidth', 2);
        semilogy(1:t, mse_test(1:t), 'r', 'LineWidth', 2);
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
        
mse_train_result = mse_train(epoch);
mse_test_result = mse_test(epoch);
mse_validation_result = mse_validation(epoch);




