function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % create a copy of theta for simultaneous update.
    % theta_n = theta;

    % number of features.
    p = size(X, 2);

    for j = 1:p
        % calculate dJ/d(theta_j)
        %deriv = ((X*theta_n - y)' * X(:, j))/m;

        % % update theta_j
        %theta(j) = theta_n(j) - (alpha*deriv);
        
        h = X * theta; % hypothesis vector
        % t1 = theta(1) - alpha * (1/m) * sum(h - y);
        t = theta(j) - alpha * (1/m) * sum((h - y) .* X(:, j));
        theta(j) = t;
        %theta(2) = t2;
    end









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
