function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predictions = X*theta;
sqrErrors = (predictions - y).^2;
J = 1/(2*m) * sum(sqrErrors);

h = X*theta;
error = h - y;
grad = X'*error;
grad = grad/m;

theta(1) = 0;
theta_squared_sum = sum(theta.^2);
scaling = lambda/(2*m);
theta_squared_sum = theta_squared_sum*scaling;
J = J + theta_squared_sum;

scaling_grad = lambda/m;
grad_reg = theta*scaling_grad;
grad = grad + grad_reg;









% =========================================================================

grad = grad(:);

end
