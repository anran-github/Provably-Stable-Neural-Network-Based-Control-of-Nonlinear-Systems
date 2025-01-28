%% This is the new system: Inverted Pedulum system

clc
close all
clear all


%%  Basic Settings
% theta
theta = 1;
% Save the data to a text file
filenames = ["drone_001_x.csv","drone_001_y.csv","drone_001_z.csv"];


%% System settings

alpha_x = 0.0527;
alpha_y = 0.0187;
alpha_z = 1.7873;
alpha = [alpha_x,alpha_y,alpha_z];

beta_x = -5.4779;
beta_y = -7.0608;
beta_z = -1.7382;
beta = [beta_x,beta_y,beta_z];

%% Loop starts

% [column1:x1, x2 col2-3:P, col4: K]


for direction=3:3
    filename = filenames(direction);
    count = 1;
    steps = 0.01;

    % read A,B,C,D matrices:
    A = [0 1 ; 0 -alpha(direction)];
    B=[0;beta(direction)];
    C = [1 0];
    D=0;
    G=ss(A,B,C,D);
    
    Gd=c2d(G,0.1);
    Ad=Gd.A;
    Bd=Gd.B;
    
    % reference pose
    x_r = [0;0];
    ranges = [-0.5:steps:0.5];
    if direction == 3
        x_r = [1.5;0];
        ranges = [1:steps:2];
    end

    

    
    for px = ranges
        for px_dot = -1:steps:1
            % current state
            x = [px;px_dot];
            % variables to optimize
            u = sdpvar(1,1,'full');
            P=sdpvar(2,2,'symmetric');
            
            Objective = 0.1*u'*u+ (Ad*x+Bd*u-x_r)'*[20 0;0 0.1]*(Ad*x+Bd*u-x_r)+((x-x_r)'*P*(x-x_r));
            Constraints = [P>=1e-10;
            ((x-x_r)'*P*(x-x_r))>=((1.5*theta)^2)*(x-x_r)'*(x-x_r);
            ((Ad*x+Bd*u-x_r)'*P*(Ad*x+Bd*u-x_r))<=((0.5*theta)^2)*(x-x_r)'*(x-x_r)];
        
            % opt=sdpsettings('solver','bmibnb');
            % opt=sdpsettings('solver','fmincon','MaxIter',2000);
            sol=optimize(Constraints,Objective)
                    
            
            chache_matrix(:,1) = x;
            p = double(P);
            % eig(p)
            chache_matrix(:,2:3) = p;
            chache_matrix(1,4) = double(u);
            chache_matrix(2,4) = x_r(1,1);
            result_matrix(:,:,count) = chache_matrix;
            count = count+1;
        
                
            % Clear variables
            clear('yalmip')
       
        end
    end
    % save result matrix and clear it. recount numbers
    writematrix(result_matrix, filename,'WriteMode','append');
    clear result_matrix
    count = 1;

end

