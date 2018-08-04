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
n = length(grad);

one = 0;
zero = 0;

for i = 1:m
  if y(i) == 1
    J = J + log(sigmoid(X(i,:) * theta));
    one = one + 1;
  else
    J = J + log(1 - sigmoid(X(i,:) * theta));
    zero = zero + 1;
  end
  
end
J = -1 * J / m ;

for k = 1:n
  for i = 1:m
    grad(k) = grad(k) + (sigmoid(X(i,:) * theta) - y(i)) * X(i,k);
    %grad(2) = grad(2) + ((1 / (1+exp(-(theta(1) * x1(i) + theta(2) * x2(i) + theta(3) * x3(i))))) - y(i)) * x2(i);
    %grad(3) = grad(3) + ((1 / (1+exp(-(theta(1) * x1(i) + theta(2) * x2(i) + theta(3) * x3(i))))) - y(i)) * x3(i);
  end
end

grad = grad / m;







% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
