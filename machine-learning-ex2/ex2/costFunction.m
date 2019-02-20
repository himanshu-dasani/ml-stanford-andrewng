function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
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
J = 1/m * sum(final_term);

error = prediction - y;
theta_change = X'*error;
grad = 1/m * theta_change;






% =============================================================

end
