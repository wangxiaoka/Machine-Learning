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

%size(theta)
%size(X)
%size(y)

n = length(grad);

for i = 1:m
  if y(i) == 1
    J = J + log(sigmoid(X(i,:) * theta)); %+ lambda * dot(theta,theta) / 2;
  else
    J = J + log(1 - sigmoid(X(i,:) * theta)); %+ lambda * dot(theta,theta) / 2;
  end
end

J_theta = 0;
for k = 2:n
  J_theta = J_theta + theta(k) * theta(k);
end
J = (-1 / m) * J + lambda * J_theta / (2*m);

  %J = (-1 / m) * (J - ((lambda * dot(theta,theta)) / 2));


for i = 1:m
  grad(1) = grad(1) + (sigmoid(X(i,:) * theta) - y(i)) * X(i,1);
end
grad(1) = grad(1) / m;


for k = 2:n
  for i = 1:m
    grad(k) = grad(k) + (sigmoid(X(i,:) * theta) - y(i)) * X(i,k);
    %grad(2) = grad(2) + ((1 / (1+exp(-(theta(1) * x1(i) + theta(2) * x2(i) + theta(3) * x3(i))))) - y(i)) * x2(i);
    %grad(3) = grad(3) + ((1 / (1+exp(-(theta(1) * x1(i) + theta(2) * x2(i) + theta(3) * x3(i))))) - y(i)) * x3(i);
  end
  grad(k) = (grad(k) + lambda * theta(k)) / m;
end


%grad(1) = grad(1) / m;
%for k = 2:n
    %grad(k) = (grad(k) + lambda * theta(k)) / m;
  %grad(3) = grad(3) / m + lambda * theta(3) / m;
%end






% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
