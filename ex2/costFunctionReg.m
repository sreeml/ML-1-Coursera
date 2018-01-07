function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


z = X * theta;
h = sigmoid(z);

left = -y' * log(h); 
right = (1-y)' * log(1-h);
C = (1/m) * (left - right);

theta(1) = 0;

SS = theta' * theta; %calculate the sum of the squares of theta.
J = C + ((lambda/(2*m)) * SS);

% Since you forced theta(1) to be zero, the grad(1) term will only be the unregularized value.
L = (1/m) * (X' * (h-y));
R = (lambda/m) * theta;
grad = L + R;




% =============================================================

end
