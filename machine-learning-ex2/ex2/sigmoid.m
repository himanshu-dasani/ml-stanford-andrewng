function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

exp = e .^ z;
exp_inv = 1 ./ exp;
one_plus = 1 + exp_inv;
g = 1 ./ one_plus;



% =============================================================

end
