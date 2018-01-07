function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Co  r or scalar).

%g(z) = 1/(1+e^-z)



g = 1 ./ (1 + e .^ -z);

% =============================================================

end
