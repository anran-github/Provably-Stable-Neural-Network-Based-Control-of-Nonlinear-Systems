clear all
close all
clc

global SIASclient    FLAG
global goalDesired Error CommandInput SS SS2 commandArray_Idf Cont GoalPt CI
global deltaT inc drone_mode ContGoal xyDesired vertDesired yawDesired DESIREDPOINT
global p state state2 xyDesired vertDesired rollDesired pitchDesired yawDesired DESIRED
global SIASclient Error CommandInput ContGoal Time netx nety netz
global goalDesired SS SS2 commandArray_Idf CI
global deltaT drone_mode Cont GoalPt DESIRED1
global p state state2 inc goalDesired deltaT ...
    Kx Ky Kz INIT_POS_ADJUST x_init...
    xyDesired vertDesired rollDesired pitchDesired yawDesired DESIREDPOINT
 

%% State space matrices.

deltaT=0.1;

alpha_x = 0.0527;
alpha_y = 0.0187;
alpha_z = 1.7873;
alpha = [alpha_x,alpha_y,alpha_z];

beta_x = -5.4779;
beta_y = -7.0608;
beta_z = -1.7382;
beta = [beta_x,beta_y,beta_z];


%% LQR and REFERENCE settings
x_init = [[0.5;0],[0.5;0],[1.5;0]];

x_r = [[0;0],[0;0],[1.5;0]];

INIT_POS_ADJUST = 1;
Q = [8 0;0 0.1];
R = 0.1;
Qz = [20 0;0 0.1];
C = [1 0];
D=0;

% X-direction
Ax = [0 1 ; 0 -alpha(1)];
Bx=[0;beta(1)];
Gx = ss(Ax,Bx,C,D);
Gxd = c2d(Gx,deltaT);
Axd=Gxd.A;
Bxd=Gxd.B;

Ay = [0 1 ; 0 -alpha(2)];
By=[0;beta(2)];
Gy = ss(Ay,By,C,D);
Gyd = c2d(Gy,deltaT);
Ayd=Gyd.A;
Byd=Gyd.B;

Az = [0 1 ; 0 -alpha(3)];
Bz=[0;beta(3)];
Gz = ss(Az,Bz,C,D);
Gzd = c2d(Gz,deltaT);
Azd=Gzd.A;
Bzd=Gzd.B;

% GET FEEDBACK GAINS
[Kx,Sx,CLP]=dlqr(Axd,Bxd,Q,R);
[Ky,Sy,CLP]=dlqr(Ayd,Byd,Q,R);
[Kz,Sz,CLP]=dlqr(Azd,Bzd,Qz,R);

%% LOAD NN MODELS
% Load ONNX model--X
netx = importNetworkFromONNX("weights_drone_weight_best/1_x/NN_x.onnx");
% model = importONNXNetwork('model.onnx');

netx.Layers(1).InputInformation;

X = dlarray(rand(1, 3), 'UU');
netx = initialize(netx, X);
summary(netx)


% Load ONNX model--Y
nety = importNetworkFromONNX("weights_drone_weight_best/1_y/NN_y.onnx");
% model = importONNXNetwork('model.onnx');

nety.Layers(1).InputInformation;

nety = initialize(nety, X);
summary(nety)

% Load ONNX model--X
netz = importNetworkFromONNX("weights_drone_weight_best/1_z/NN_z.onnx");
% model = importONNXNetwork('model.onnx');

netz.Layers(1).InputInformation;

netz = initialize(netz, X);
summary(netz)





%%

SIASclient = natnet;
SIASclient.connect; 
pause(2)

fprintf('\n\nConnecting to Drone...\n') 
p = parrot(); 
fprintf('Connected to %s\n', p.ID) 
fprintf('Battery Level is %d%%\n', p.BatteryLevel)
takeoff(p); 
pause(1)







 
 Error=zeros(1,1);
 CommandInput=zeros(4,1);
inc=1;


SamplingTime = timer('ExecutionMode','fixedRate','Period',deltaT,'TimerFcn',@(~,~)myfile);
start(SamplingTime);


function myfile
% while(1>0)
global SIASclient Error TimeGlob TG  xhat_p FLAG Time  SS deltaT commandArray_Idf CI  DESIRED1 state  inc p   xyDesired vertDesired   yawDesired r GoalPt goalDesired deltaT


Time(inc,1)=double(SIASclient.getFrame.fTimestamp);

    Position=double([SIASclient.getFrame.RigidBodies(1).x;SIASclient.getFrame.RigidBodies(1).y;SIASclient.getFrame.RigidBodies(1).z]);
    q=quaternion( SIASclient.getFrame.RigidBodies(1).qw, SIASclient.getFrame.RigidBodies(1).qx, SIASclient.getFrame.RigidBodies(1).qy, SIASclient.getFrame.RigidBodies(1).qz );
    eulerAngles=quat2eul(q,'xyz')*180/pi;
    Angle=[eulerAngles(1);eulerAngles(2);eulerAngles(3)];
    state=[Position;Angle];
    [errorArray]=ControlCommand;

        SS(:,inc)=state;
        CI(:,inc)=commandArray_Idf;
        Error(:,inc)=errorArray;
        inc=inc+1;

% end

end


function [errorArray]=ControlCommand
 tic
global p state  inc FLAG SS CI Error commandArray_Idf  ...
GoalPt ContGoal goalDesired  Time DESIRED netx nety netz ...
Kx Ky Kz x_r x_init INIT_POS_ADJUST deltaT


%% Define desired tolerances and gains





if inc==1
    example_x = dlarray([state(1),(state(1))/deltaT,0], 'UU');
else
example_x = dlarray([state(1),(state(1)-SS(1,inc-1))/deltaT,0], 'UU');
end

% Perform inference
output_x = predict(netx, example_x);
ux = extractdata(output_x(4));


if ux>=0.05
    ux=0.05;
elseif ux<=-0.05
    ux=-0.05;
end

if inc==1
    example_y = dlarray([state(2),(state(2))/deltaT,0], 'UU');
else
    example_y = dlarray([state(2),(state(2)-SS(2,inc-1))/deltaT,0], 'UU');
end

% Perform inference
output_y = predict(nety, example_y);
uy = extractdata(output_y(4));

if uy>=0.05
    uy=0.05;
elseif uy<=-0.05
    uy=-0.05;
end


if inc==1
    example_z = dlarray([state(3),(state(3))/deltaT,1.5], 'UU');
else
    example_z = dlarray([state(3),(state(3)-SS(3,inc-1))/deltaT,1.5], 'UU');
end

% Perform inference
output_z = predict(netz, example_z);
uz = extractdata(output_z(4));

if uz>=0.5
    uz=0.5;
elseif uz<=-0.5
    uz=-0.5;
end



%#####################LQR initial state adjust:#########################
thresh_error = 0.1;
if INIT_POS_ADJUST > 100
    if inc==1
        x = [state(1);(state(1))/deltaT];
        y = [state(2);(state(2))/deltaT];
        z = [state(3);(state(3))/deltaT];
    else
        x = [state(1);(state(1)-SS(1,inc-1))/deltaT];
        y = [state(2);(state(2)-SS(2,inc-1))/deltaT];
        z = [state(3);(state(3)-SS(3,inc-1))/deltaT];
    end
    % adjust init position
    ux = -Kx*(x-x_init(:,1));
    uy = -Ky*(y-x_init(:,2));
    uz = -Kz*(z-x_init(:,3));
    % uz=0;
    
    if uz>=0.5
        uz=0.5;
    elseif uz<=-0.5
        uz=-0.5;
    end
    if uy>=0.05
        uy=0.05;
    elseif uy<=-0.05
        uy=-0.05;
    end
    if ux>=0.05
        ux=0.05;
    elseif ux<=-0.05
        ux=-0.05;
    end
    init_error = norm(state(1:3)-x_init(1,:)');
    % wait unit error is less than thresh_error enough.
    if init_error < thresh_error
        INIT_POS_ADJUST = INIT_POS_ADJUST + 1;
    end
end
%##################### LQR initial state adjust END ####################


 if inc == 1 
    old_yaw_Error = 0; 
 else
     old_yaw_Error=Error(1,inc-1);

 end


 yawActual = deg2rad(state(6)); 


 
 % to rotate X,Y from world frame to robot frame
 Tw2r = [cos(yawActual), sin(yawActual); -sin(yawActual), cos(yawActual)]; 


 % 
 
 
 
 % Compute the errors
 % Yaw Error


yawe1=(deg2rad(0) - yawActual);
 yawError = wrapToPi(yawe1);
 yawD_Error = (yawError-old_yaw_Error)/deltaT;
 
kYaw =-0.2; 
kD_Yaw =-0.2;

 % compute the yaw commands
 yawCmd = kYaw*yawError+kD_Yaw*yawD_Error;
 
 if abs(yawCmd) > 3.4 
 yawCmd = sign(yawCmd)*3.4; 
 end 



 errorArray = [yawError]; 


  % if inc>1 
  %     if Time(inc,1)-Time(inc-1,1)>eps
  %         commandArray_Idf= [ux; uy; yawCmd; uz];
  %         FLAG(inc)=1;
  % else
  %     commandArray_Idf= [CI(1,inc-1); CI(2,inc-1); CI(3,inc-1); CI(4,inc-1)];
  %     FLAG(inc)=0;
  %     end
  % else
  %     commandArray_Idf= [ux; uy; yawCmd; uz];
  %      FLAG(inc)=1;
  % end

commandArray_Idf= [ux; uy; yawCmd; uz];

move(p, 1.1*deltaT, 'RotationSpeed', commandArray_Idf(3),'VerticalSpeed', commandArray_Idf(4),'roll', commandArray_Idf(2), 'pitch', commandArray_Idf(1));

toc



end


