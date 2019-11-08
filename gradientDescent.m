function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%   GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples

J_history = zeros(num_iters, 1);  % Return a matrix or N-dimensional array 
                                  % whose elements are all 0.
                                  % If invoked with a single scalar integer 
                                  % argument, return a square NxN matrix.

for iter = 1:num_iters

    % You can check the pdf i made, page 23, the picture of the slide
    % express very well what we do here
    
    
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
