close all
clear all

%% System settings

alpha_x = 0.0527;
alpha_y = 0.0187;
alpha_z = 1.7873;
alpha = [alpha_x,alpha_y,alpha_z];

beta_x = -5.4779;
beta_y = -7.0608;
beta_z = -1.7382;
beta = [beta_x,beta_y,beta_z];

%% LQR settings
Q = [20 0;0 0.1];
R = 0.1;

C = [1 0];
D=0;

% X-direction
Ax = [0 1 ; 0 -alpha(1)];
Bx=[0;beta(1)];
Gx = ss(Ax,Bx,C,D);
Gxd = c2d(Gx,0.1);
Axd=Gxd.A;
Bxd=Gxd.B;

Ay = [0 1 ; 0 -alpha(2)];
By=[0;beta(2)];
Gy = ss(Ay,By,C,D);
Gyd = c2d(Gy,0.1);
Ayd=Gyd.A;
Byd=Gyd.B;

Az = [0 1 ; 0 -alpha(3)];
Bz=[0;beta(3)];
Gz = ss(Ax,Bx,C,D);
Gzd = c2d(Gz,0.1);
Azd=Gzd.A;
Bzd=Gzd.B;

%% LOOPING Simulation

% [Kx,Sx,CLP]=lqr(Gx,Q,R)
% % u=-K*x
% x_tt = xt + Ax*xt +Bx*(-kx*xt)


x1 = 1.;
x2 = 0.5;
r = 0;

% init point and ref for z direction.
x1 = 1.;
x2 = -0.5;
r = 1.5;
[Kx,Sx,CLP]=dlqr(Azd,Bzd,Q,R);

x_r = [r;0];
% given a random point, go to reference r. with num_compare steps.
p_tt = eye(2);
dt = 0.01;
tStart = tic; 
cnt = 1;
num_compare = 500;

% Preallocate arrays for storing states and control inputs
X = zeros(num_compare, 2);  % States
U = zeros(num_compare, 1);  % Control inputs


while cnt <= num_compare
    
    % OPtimization part
    if cnt == 1
        x = [x1;x2];
    else
        x = xtt;
    end
             
    

    
    u = -Kx*(x-x_r);

    xtt =  Azd*x +Bzd*u;
    
    % Store state and control input
    X(cnt, :) = x';
    U(cnt, :) = u';
    cnt = cnt + 1;

end


elapsed_time = toc;

disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);
% writematrix(opt_result, save_name);
%%  display
t = 0:dt:(num_compare-1)*dt;
% Plot results
figure;
subplot(2, 1, 1);
plot(t, X(:, 1), 'b', 'LineWidth', 2);
hold on;
plot(t, r * ones(size(t)), 'r--', 'LineWidth', 2);  % Plot reference
xlabel('Time');
ylabel('State (x1)');
legend('Actual', 'Reference');
title('State Response');

subplot(2, 1, 2);
plot(t, U, 'g', 'LineWidth', 2);
xlabel('Time');
ylabel('Control Input (u)');
title('Control Input');

figure(2)
plot(X(:,1),X(:,2),'-+')
xlabel('x1 [m]')
ylabel('x2 [m/s]')
grid("on")
title('trajectory with given random points')
