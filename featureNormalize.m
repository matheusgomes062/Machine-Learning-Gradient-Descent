function [X_norm, mu, sigma] = featureNormalize(X)
%   FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% mu
for p = 1:size(X, 2)
  mu(p) = mean(X(:, p), "a");
end

% sigma
for p = 1:size(X, 2)
  sigma(p) = std(X(:, p));
end

% X_norm
for p = 1:size(X, 2)
  if (sigma(p) != 0)
    for i = 1:size(X, 1)
      X_norm(i, p) = (X(i, p)-mu(p))/sigma(p);
    end
  else
    % sigma(p) == 0 <=> forall i, j,  X(i, p) == X(j, p) == mu(p)
    % In this case,  normalized values are all zero.
    % (mean is 0,  standard deviation is sigma(=0))
    X_norm(:, p) = zeros(size(X, 1), 1);
  end
end







% ============================================================

end
