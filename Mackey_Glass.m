clear; clc;
%% Mackey Glass

tmax = 1100; %number of data
tau = 17;    
n = 10;      %pow
a = 0.1;
b = 0.2;
x = zeros(17,1);
x(18,1) = 1.2;
for i = 18:(tmax-1)
    x(i+1) = x(i) - a * x(i) + ( (b * x(i-tau) ) / ( 1 + x(i-tau)^n) );
end

% Normalizing
min = min(x);
max = max(x);

x_norm = (x - min) ./ max;

Delay_x = [x_norm(1:end-3),x_norm(2:end-2),x_norm(3:end-1),x_norm(4:end)];

% Genarate Weights
num_data = size(Delay_x,1);
data = [ones(num_data,1) Delay_x];  %data with bias input

num_input = size(data,2) - 1;

n1 = num_input; % number of input
n2 = 5;         % number of neurons in the hidden layer
n3 = 1;         % number of neurons in the output layer

rng(42);
a = -1;
b = 1;
w1 = unifrnd(a, b, [n2 n1]);
w2 = unifrnd(a, b, [n3 n2]);

%% PLOT

figure()
plot(x,'b','linewidth',1);title('Mackey Glass');xlabel('time');ylabel('x(t)');

%% Save

csvwrite('normalize_dataset.csv',Delay_x);
csvwrite('mackey_glass_data.csv',x);
csvwrite('initial_w1.csv',w1);
csvwrite('initial_w2.csv',w2);
