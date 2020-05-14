function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z.

g = zeros(size(z));

g = sigmoid(z) .* (1 - sigmoid(z));		% sigmoid(z)(m,1) g(10,1)

end;
