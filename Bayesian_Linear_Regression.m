%=============================
% Author: Antonis Kastellakis
%=============================

clear, clc, close all

% Load our dataset.
data = importdata('data.txt') ; 
X_data = data(:,1);
N = length(X_data);
t = data(:,2);

%=====
% (a)
%=====
% Set constant values.
% Reminder: precision = 1/variance.
% alpha, precision of the prior
alpha = 1e-1;
% beta,  precision of the likelihood
beta = 5;        

%=====
% (b)
%=====
% Use two sets of basis functions, one Gaussian and one sigmoidal, 
% with exactly M = 10 functions in each case. 
M = 10;
% For the means and deviations we decide to spread them equally over
% [0,1].
mu = (0.1:0.1:1)';
sigma = (0.1:0.1:1)';

%=====
% (c)
%=====
% Create the design matrix for Gaussian basis functions
Phi_Gaussian = zeros(N, M);
for i = 1:M
    Phi_Gaussian(:, i) = exp(-(X_data - mu(i)).^2 / (2*(sigma(i)^2)));
end

% Optimal parameters alpha and beta for the Gaussian case (Bonus).
[alpha_Gaussian, beta_Gaussian] = parameter_optimization(Phi_Gaussian,t,alpha,beta);

% Gaussian basis functions.
SN_inv_Gaussian = alpha_Gaussian * eye(M) + beta_Gaussian * (Phi_Gaussian') * Phi_Gaussian;
SN_Gaussian = inv( SN_inv_Gaussian );
mN_Gaussian = beta_Gaussian * SN_Gaussian * ( Phi_Gaussian' )* t;

% Functional form of the predictive distribution using the entire dataset.
% The predictive distribution also follows a Gaussian, where the mean 
% ( mN^T * phi(x)) and variance (Equation 3.59 Bishop)
% are calculated as follows for each case.

Gausian_Distribution = @(y, mu, s) 1 / (sqrt(2*pi)*s) * exp(-1/2 * ((y-mu)/s).^2);

Phi_Gaussian_x = @(x, mu,s) exp(-(1 - mu).^2 ./ (2*(sigma.^2)));

predictive_distribution_Gaussian = @(x) Gausian_Distribution(x, ( mN_Gaussian' ) * ...
    Phi_Gaussian_x(x, mu, sigma), (1 / beta_Gaussian +  (Phi_Gaussian_x(x, mu, sigma)') ...
            * SN_Gaussian *  Phi_Gaussian_x(x, mu, sigma)));

% Create the design matrix for Sigmoidal basis functions
sigmoid = @(x)  1 ./ (1 + exp(-x));
Phi_Sigmoidal = zeros(N, M);
for i = 1:M
    Phi_Sigmoidal(:, i) = sigmoid( ( X_data - mu(i)) / sigma(i) );
end

% Optimal parameters alpha and beta for the Sigmoidal case (Bonus).
[alpha_Sigmoidal, beta_Sigmoidal] = parameter_optimization(Phi_Sigmoidal,t,alpha,beta); 

% Sigmoidal basis functions.
SN_inv_Sigmoidal = alpha_Sigmoidal * eye(M) + beta_Sigmoidal * (Phi_Sigmoidal') * Phi_Sigmoidal;
SN_Sigmoidal = inv( SN_inv_Sigmoidal );
mN_Sigmoidal = beta_Sigmoidal * SN_Sigmoidal * ( Phi_Sigmoidal' )* t;

% Functional form of the predictive distribution using the entire dataset 
% for the sigmoidal basis functions case.

Phi_Sigmoid_x = @(x, mu,s)  sigmoid( ( 1 - mu) ./ sigma );

predictive_distribution_Sigmoidal = @(x) Gausian_Distribution(x, ( mN_Sigmoidal' ) * ...
    Phi_Sigmoidal_x( x, mu, sigma ), ( 1 / beta_Sigmoidal +  (Phi_Sigmoidal_x( x, mu, sigma )') ...
            * SN_Sigmoidal *  Phi_Sigmoidal_x(x, mu, sigma)));

%=====
% (d)
%=====

% Number of data we shall use to learn.
N = [10 30];

axis = linspace(0,1);
true_f = [];

for x = linspace(0,1)
    true_f(end+1)  = cos(2*pi*x) - (3*x - 2)^2;
end

num = 1;

for n = N
    % Gaussian basis functions.
    Phi_G = Phi_Gaussian(1:n,:);
    SN_inv_G = alpha_Gaussian * eye(M) + beta_Gaussian * (Phi_G') * Phi_G;
    SN_G = inv( SN_inv_G );
    mN_G = beta_Gaussian * SN_G * ( Phi_G' )* t(1:n);
    
    predicted_mean_Gaussian = [];
    predicted_variance_Gaussian = [];
    upper_uncertainty_bound_Gaussian = [];
    lower_uncertainty_bound_Gaussian = [];
    
    % Sigmoidal basis functions.
    Phi_S = Phi_Sigmoidal(1:n,:);
    SN_inv_S = alpha_Sigmoidal * eye(M) + beta_Sigmoidal * (Phi_S') * Phi_S;
    SN_S = inv( SN_inv_S );
    mN_S = beta_Sigmoidal * SN_S * ( Phi_S' )* t(1:n);
    
    predicted_mean_Sigmoid = [];
    predicted_variance_Sigmoid = [];
    upper_uncertainty_bound_Sigmoid = [];
    lower_uncertainty_bound_Sigmoid = [];
    
    for x = linspace(0,1)
    
        phi_x_Gaussian = zeros(M,1);
        phi_x_Sigmoid = zeros(M,1);
        for i = 1:M
            phi_x_Gaussian(i) = exp( - ( x - mu(i) ).^ 2 / ( 2 * ( sigma(i) ^ 2 )));
            phi_x_Sigmoid(i) = sigmoid( ( x - mu(i)) / sigma(i) );
        end
    
        % Gaussian basis functions.
        predicted_mean_Gaussian(end+1)  = ( mN_G' ) * (phi_x_Gaussian);
        predicted_variance_Gaussian(end+1) = 1 / beta_Gaussian +  phi_x_Gaussian' ...
            * SN_G *  ( phi_x_Gaussian );
        upper_uncertainty_bound_Gaussian(end+1) = predicted_mean_Gaussian(end) ...
            + sqrt( predicted_variance_Gaussian(end));
        lower_uncertainty_bound_Gaussian(end+1) = predicted_mean_Gaussian(end) ...
            - sqrt( predicted_variance_Gaussian(end));
    
    
        % Sigmoid basis functions.
        predicted_mean_Sigmoid(end+1)  = ( mN_S' ) * (phi_x_Sigmoid);
        predicted_variance_Sigmoid(end+1) = 1 / beta_Sigmoidal +  phi_x_Sigmoid' ...
            * SN_S *  ( phi_x_Sigmoid );
        upper_uncertainty_bound_Sigmoid(end+1) = predicted_mean_Sigmoid(end) ...
            + sqrt( predicted_variance_Sigmoid(end));
        lower_uncertainty_bound_Sigmoid(end+1) = predicted_mean_Sigmoid(end) ...
            - sqrt( predicted_variance_Sigmoid(end));
    
    end
    
    % Plots that show the true function, the data points, and the mean and
    % variance of our predictive distribution in each case.
    figure(num);
    num = num +1 ;
    % uncertainty variance margins
    xconf = [axis axis(end:-1:1)] ;         
    yconf = [lower_uncertainty_bound_Gaussian upper_uncertainty_bound_Gaussian(end:-1:1)];
    p = fill(xconf,yconf,'red');
    p.FaceColor = [1 0.8 0.8];      
    p.EdgeColor = 'none';
    hold on;
    plot(axis, predicted_mean_Gaussian, 'r')
    scatter(X_data(1:n), t(1:n), 'b')
    plot(axis,true_f, 'g')
    hold off;
    str = "Gaussian basis functions with N = " + n + " data points used";
    title(str,'Interpreter','latex', 'Fontsize', 14);
    legend('Uncertainty of prediction', 'Predicted mean', 'Training Data', 'True Function',  ...
       'Location', 'best', 'Interpreter','latex', 'Fontsize', 14);
   
    figure(num);
    num = num +1 ;
    xconf = [axis axis(end:-1:1)] ;         
    yconf = [lower_uncertainty_bound_Sigmoid upper_uncertainty_bound_Sigmoid(end:-1:1)];
    p = fill(xconf,yconf,'red');
    p.FaceColor = [1 0.8 0.8];      
    p.EdgeColor = 'none';
    hold on;
    plot(axis, predicted_mean_Sigmoid, 'r')
    scatter(X_data(1:n), t(1:n), 'b')
    plot(axis, true_f, 'g')
    hold off;
    str = "Sigmoidal basis functions with N = " + n + " data points used";
    title(str, 'Interpreter','latex', 'Fontsize', 14);
    legend('Uncertainty of prediction', 'Predicted mean', 'Training Data', 'True Function',  ...
        'Location', 'best', 'Interpreter','latex', 'Fontsize', 14);
end

%=====  
% (e)
%=====

for n = N
    % Gaussian basis functions.
    Phi_G = Phi_Gaussian(1:n,:);
    SN_inv_G = alpha_Gaussian * eye(M) + beta_Gaussian * (Phi_G') * Phi_G;
    SN_G = inv( SN_inv_G );
    mN_G = beta_Gaussian * SN_G * ( Phi_G' )* t(1:n);
    
    % Plots that show the true function, the data points, and the five sampled
    % regression functions in each case.
    figure(num);
    num = num +1 ;
    scatter(X_data(1:n), t(1:n), 'b')
    hold on;
    plot(axis,true_f, 'g')
    
    % Sample 5 regression functions by drawing their parameters w from
    % the posterior distrubition of the weights.
    % This distribution is also a Gaussian with mN as mean and SN as
    % covariance.    
    for k = 1 : 5
        w_G = randn(1,M) * chol(SN_G) + mN_G'; % mvnrnd(mN_G,SN_G)
        y_G = [];
        for x = linspace(0,1)
            phi_x_Gaussian = zeros(M,1);
            for i = 1:M
                phi_x_Gaussian(i) = exp( - ( x - mu(i) ).^ 2 / ( 2 * ( sigma(i) ^ 2 )));
            end
            y_G(end+1)= w_G * phi_x_Gaussian;
        end  
        plot(axis,y_G)
    end 
    
    hold off;
    str = "Sampled Regression functions with N = " + n + " data (Gaussian)";
    title(str,'Interpreter','latex', 'Fontsize', 14);
   
    % Sigmoidal basis functions.
    Phi_S = Phi_Sigmoidal(1:n,:);
    SN_inv_S = alpha_Sigmoidal * eye(M) + beta_Sigmoidal * (Phi_S') * Phi_S;
    SN_S = inv( SN_inv_S );
    mN_S = beta_Sigmoidal * SN_S * ( Phi_S' )* t(1:n);
   
    % Plots that show the true function, the data points, and the five sampled
    % regression functions in each case.
    figure(num);
    num = num +1 ;
    scatter(X_data(1:n), t(1:n), 'b')
    hold on;
    plot(axis,true_f, 'g')
    
    % Sample 5 regression functions by drawing their parameters w from
    % the posterior distrubition of the weights.
    % This distribution is also a Gaussian with mN as mean and SN as
    % covariance.    
    for k = 1 : 5
        w_S = randn(1,M) * chol(SN_S) + mN_S' ; % mvnrnd(mN_S,SN_S)
        y_S = [];
        for x = linspace(0,1)
            phi_x_Sigmoid = zeros(M,1);
            for i = 1:M
                phi_x_Sigmoid(i) = sigmoid( ( x - mu(i)) / sigma(i) );
            end
            y_S(end+1)= w_S * phi_x_Sigmoid;
        end  
        plot(axis,y_S)
    end 
    
    hold off;
    str = "Sampled Regression functions with N = " + n + " data (Sigmoid)";
    title(str,'Interpreter','latex', 'Fontsize', 14);
end