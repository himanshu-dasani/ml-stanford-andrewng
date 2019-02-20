function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

prediction = X*Theta';
movie_rating_error = prediction - Y;
error_factor = movie_rating_error .* R;
error_factor_squared = error_factor .* error_factor;
error_sum = sum(error_factor_squared(:));
J = (1/2)*error_sum; 

X_grad = error_factor*Theta;
Theta_grad = error_factor'*X;

X_squared = X .* X;
X_squared_sum = sum(X_squared(:));
X_squared_sum = (lambda/2)*X_squared_sum;
Theta_squared = Theta .* Theta;
Theta_squared_sum = sum(Theta_squared(:));
Theta_squared_sum = (lambda/2)*Theta_squared_sum;

J = J + X_squared_sum + Theta_squared_sum;

X_grad_reg = X*lambda;
Theta_grad_reg = Theta*lambda;

X_grad = X_grad + X_grad_reg;
Theta_grad = Theta_grad + Theta_grad_reg;














% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end