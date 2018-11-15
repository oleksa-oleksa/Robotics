% Robotics FU
% Assignment 2

% Rotate a vector (2, 0, 0) using quaternion rotation by -3π/2 around axis (0,0,1), 
% what is the resulting vector? Provide the calculation steps.

theta = (-3 * pi) / 2

K = [0 0 1]

E0_1 = cos(theta/2);
E1_1 = K(1) * sin(theta/2);
E2_1 = K(2) * sin(theta/2);
E3_1 = K(3) * sin(theta/2);

quat = [E0_1 E1_1 E2_1 E3_1]

% Derive Angle and Axis from Quaternion

E0 = 0.5;
E1 = -0.5;
E2 = -0.5;
E3 = 0.5;

theta_rad = 2 * acos(E0)
theta_grad = (2 * acos(E0)) * (180/pi)

V_x = E1 /(sin(theta_rad/2))
V_y = E2 /(sin(theta_rad/2))
V_z = E3 /(sin(theta_rad/2))





