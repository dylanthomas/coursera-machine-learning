function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

params = [];

for C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        params = [ params ; C, sigma, mean(double(predictions ~= yval))];
    end
end

fprintf('# Parameters Selection Results\n');
fprintf('# \tC\t\tsigma\t\tError\n');
for i = 1:size(params, 1)
    fprintf('  \t%f\t%f\t%f\n', params(i, 1), params(i, 2), params(i, 3));
end


[val,index] = min(params(:, 3));

fprintf('# \tMinimum Error\t\tIndex\t\tC\t\tsigma\n');
fprintf('  \t%f\t\t%d\t\t%f\t%f\n', val, index, params(index, 1), params(index, 2));

C = params(index, 1);
sigma = params(index, 2);



% =========================================================================

end
