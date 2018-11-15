% Robotics FU
% Assignment 2

% Calculate Z Axis

x = [-sqrt(0.5) sqrt(0.5) 0]; 
y = [sqrt(0.5) sqrt(0.5) 0];

% The cross product or vector product: given two linearly independent vectors a and b , 
% the cross product, is a vector that is perpendicular to both a and b

z = cross(x, y)