function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


z = X*theta;
prediction = sigmoid(z);
prediction_logged = log(prediction);
minus_onned = -1 .* y;
term_one = minus_onned .* prediction_logged;
one_minus = 1 - y;
one_minus_prediction = 1 - prediction;
prediction_logged_one_minus = log(one_minus_prediction);
term_two = one_minus .* prediction_logged_one_minus;
final_term = term_one - term_two;
J_one = 1/m * sum(final_term);

theta(1) = 0;
theta_sum_of_sq = theta' * theta;
cons = lambda/(2*m) ;
J_two = cons * theta_sum_of_sq;
J = J_one + J_two;

error = prediction - y;
theta_change = X'*error;
grad_one = 1/m * theta_change;

grad_two = (lambda/m) * theta;

grad = grad_one + grad_two;








% =============================================================

grad = grad(:);

end
