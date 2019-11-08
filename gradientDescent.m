function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
% J_history = zeros(num_iters + 1, 1);  % Return a matrix or N-dimensional array 
                                  % whose elements are all 0.
                                  % If invoked with a single scalar integer 
                                  % argument, return a square NxN matrix.
                                  % In this case we are returning a num_iters + 1 X 1
                                  % matrix. It is + 1 because of the first 
                                  % variable that is 1
J_history = zeros(num_iters, 1);
                                  
% m = length(y); % number of training examples
% J_history = zeros(num_iters+1, 1);
% theta_history = zeros(num_iters + 1, size(theta', 2)); % adding ones column to X
% theta_history(1, :) = theta';
% J_history(1)= computeCost(X, y, theta);
                                  
% for iter = 1:num_iters + 1
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % You can check the pdf i made, page 23, the picture of the slide
    % express very well what we do here
    % theta = theta - ((alpha / m) * (X' * (X * theta - y)));

    % create a copy of theta for simultaneous update.
    % It's important to make it simultaneous as stated in the video.
    % theta_prev = theta;

    % number of features.
    % p = size(X, 2);

    % simultaneous update theta using theta_prev.
    % here i have one doubt, should'nt it be from 1 to m? (1:m) 
    % for j = 1:p
    % (exactly the same with multivariate version)
    % Here we have exactly the gradiant descent algorithm showed in week 1
    % Parameter learning, Gradiant Descent for Linear Regression
    % The only exception is the m that is in the deriv variable of the first 
    % calculation, but it could be along side the second, as showed in the slide.
    %    deriv = ((X * theta_prev - y)' * X(:, j))/m;

        % update theta_j
    %    theta(j) = theta_prev(j)-(alpha * deriv);
    
    h = X * theta; % hypothesis vector
    t1 = theta(1) - alpha * (1/m) * sum(h - y);
    t2 = theta(2) - alpha * (1/m) * sum((h - y) .* X(:, 2));
    theta(1) = t1;
    theta(2) = t2;
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
