%=====================================
% Simulation: 
% Aftering transfering NN model in 
% MATLAB format, verify models 
% availbility with a given random point.
%=====================================

clc
clear all


%% theta Parameter settings

% Load ONNX model
net = importNetworkFromONNX("weights_drone_weight_best/1_y/NN_y.onnx");
% model = importONNXNetwork('model.onnx');

net.Layers(1).InputInformation;

X = dlarray(rand(1, 3), 'UU');
net = initialize(net, X);
summary(net)


theta = 1;
direction = 3;
save_name = strcat('trajectory_opt_drone_theta',num2str(theta),'.csv');

alpha_x = 0.0527;
alpha_y = 0.0187;
alpha_z = 1.7873;
alpha = [alpha_x,alpha_y,alpha_z];

beta_x = -5.4779;
beta_y = -7.0608;
beta_z = -1.7382;
beta = [beta_x,beta_y,beta_z];

num_compare = 100;
result_p = zeros(1,2,num_compare);
result_u = zeros(1,num_compare);
cnt = 1;

% read A,B,C,D matrices:
A = [0 1 ; 0 -alpha(direction)];
B=[0;beta(direction)];
C = [1 0];
D=0;
G=ss(A,B,C,D);

Gd=c2d(G,0.1);
Ad=Gd.A;
Bd=Gd.B;

x1 = 1.;
x2 = 0.5;
r = 0;

% init point and ref for z direction.
% x1 = 1.55;
% x2 = -0.5;
% r = 1.5;


x_r = [r;0];
% given a random point, go to reference r. with num_compare steps.
p_tt = eye(2);
tic; 
count = 1;
while cnt <= num_compare
    
    % OPtimization part
    if cnt == 1
        x = [x1;x2];
    else
        x = xtt;
    end
 
    example_x = dlarray([x(1,1),x(2,1),r], 'UU');
    
    % Perform inference
    output = predict(net, example_x);
    
    % Display the output (adjust as per your model's output)
    disp('Model Output:');
    disp(output);
    % construct p and u with type double
    nn_p = double([output(1),output(2);output(2),output(3)]);
    opt_p = extractdata(nn_p);
    opt_u = extractdata(output(4));            
    
    opt_p    


    xtt = Ad*x +Bd*opt_u;

    % save data to compare
    % saving matrix looks like: 
    % ================================
    %   x1  delta_x1 norm(x) u delta_v  
    %   x2  delta_x2   0     0    0
    % ================================
    delta_x = xtt-x;
    delta_v = sqrt(xtt'*opt_p*xtt) - sqrt(x'*p_tt*x);
    norm_xt = norm(x);    
    % opt_result(:,:,cnt) = opt_p;
    opt_result(1,4,cnt) = opt_u;
    opt_result(1:2,1,cnt) = x;
    opt_result(1:2,2,cnt) = delta_x;
    opt_result(1,5,cnt) = delta_v;
    opt_result(1,3,cnt) = norm_xt;

    result_input(cnt,:) = x;

    % next x(t+1)
    % xtt = [x(2);-9.8*sin(x(1))] +[0;1]*opt_u;

    cnt = cnt + 1;
    p_tt = opt_p;

end


elapsed_time = toc;

disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);
% writematrix(opt_result, save_name);
%%  display
figure(1)
plot(result_input(:,1),result_input(:,2),'-+')
xlabel('x1 [m]')
ylabel('x2 [m/s]')
grid("on")
title('trajectory with given random points')

figure(2)
plot(1:num_compare,reshape(opt_result(1,5,:),1,num_compare))
xlabel('count')
ylabel('\Delta V')
grid("on")
title('Derivative of Lyapunov Function')