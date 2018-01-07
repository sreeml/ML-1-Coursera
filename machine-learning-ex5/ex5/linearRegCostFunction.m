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
%unreg cost function
h = X * theta;
error = h - y;
error_sqr = (h-y).^2;

C = 1/(2*m) * sum([error_sqr]);

%unreg gradient
#(ignore the 'alpha' variable, it is not used in this exercise). That gives us the gradient. Since we let fmincg() perform gradient descent for us, we just have to compute the cost and gradient. We don't use a for-loop over the number of iterations, or use any learning rate. The fmincg() function does that for us.

h = X * theta;
errors = h - y;
theta_change = (1/m)*(X'*errors); %change in theta or gradient,%vector multiplication takes care of sum automatically 

%reg cost function
theta(1) = 0;
SS = theta' * theta; %calculate the sum of the squares of theta OR sum(theta.^2)
J = C + ((lambda/(2*m)) * SS);

%reg gradient
R = (lambda/m) * theta;
grad = theta_change + R;










% =========================================================================

grad = grad(:);

end
