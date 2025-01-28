clear all
close all
clc

global SIASclient    FLAG
global goalDesired Error CommandInput SS SS2 commandArray_Idf Cont GoalPt CI
global deltaT inc drone_mode ContGoal xyDesired vertDesired yawDesired DESIREDPOINT
global p state state2 xyDesired vertDesired rollDesired pitchDesired yawDesired DESIRED
global SIASclient Error CommandInput ContGoal Time netx nety netz
global goalDesired SS SS2 commandArray_Idf CI
global deltaT drone_mode Cont GoalPt DESIRED1 Kx Ky Kz x_r x_init INIT_POS_ADJUST
global p state state2 inc goalDesired deltaT  xyDesired vertDesired rollDesired pitchDesired yawDesired DESIREDPOINT
 

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
x_init = [[0.8;0],[1;0],[2;0]];
% variable x_r is applied only in LQR at this moment.
x_r = [[0;0],[0;0],[1.5;0]];
INIT_POS_ADJUST = 1;
Q = [8 0;0 0.1];
R = 0.1;
Qz = [10 0;0 0.1];
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







 
 Error=zeros(4,1);
 CommandInput=zeros(4,1);
inc=1;


while(1>0)


Time(inc,1)=double(SIASclient.getFrame.fTimestamp);

    Position=double([SIASclient.getFrame.RigidBodies(1).x;SIASclient.getFrame.RigidBodies(1).y;SIASclient.getFrame.RigidBodies(1).z]);
    q=quaternion( SIASclient.getFrame.RigidBodies(1).qw, SIASclient.getFrame.RigidBodies(1).qx, SIASclient.getFrame.RigidBodies(1).qy, SIASclient.getFrame.RigidBodies(1).qz );
    eulerAngles=quat2eul(q,'xyz')*180/pi;
    Angle=[eulerAngles(1);eulerAngles(2);eulerAngles(3)];
    state=[Position;Angle];
    [errorArray]=ControlCommand;
    if FLAG(inc)==1
        SS(:,inc)=state;
        CI(:,inc)=commandArray_Idf;
        Error(:,inc)=errorArray;
        inc=inc+1;
    end
% end

end


function [errorArray]=ControlCommand
 
global p state  inc FLAG SS CI Error ...
    Kx Ky Kz x_init x_r INIT_POS_ADJUST...
    commandArray_Idf  GoalPt ContGoal goalDesired  Time DESIRED netx nety netz


%% Define desired tolerances and gains



 if inc==1
     dt=0.008301;
 else
     dt=Time(inc,1)-Time(inc-1,1);
 end

if dt < eps 
   dt = 0.008301; % Average calculated time step
end

% LQR inputs

if inc==1
    x = [state(1);0];
    y = [state(2);0];
    z = [state(3);0];
else
    x = [state(1);(state(1)-SS(1,inc-1))/dt];
    y = [state(2);(state(2)-SS(2,inc-1))/dt];
    z = [state(3);(state(3)-SS(3,inc-1))/dt];
end

% adjust init position
    ux = -Kx*(x-x_r(:,1));
    uy = -Ky*(y-x_r(:,2));
    uz = -Kz*(z-x_r(:,3));


%#####################LQR initial state adjust:#########################
thresh_error = 0.01;
if INIT_POS_ADJUST < 200
    % adjust init position
    ux = -Kx*(x-x_init(:,1));
    uy = -Ky*(y-x_init(:,2));
    uz = -Kz*(z-x_init(:,3));
    % uz=0;
    
    init_error = norm(state(1:3)-x_init(1,:)');
    % wait unit error is less than thresh_error enough.
    if init_error < thresh_error
        INIT_POS_ADJUST = INIT_POS_ADJUST + 1;
    end
end
%##################### LQR initial state adjust END ####################

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
 if inc==1
     dt=0.008301;
 else
     dt=Time(inc,1)-Time(inc-1,1);
 end

if dt < eps 
   dt = 0.008301; % Average calculated time step
end

% dt

yawe1=(deg2rad(0) - yawActual);
 yawError = wrapToPi(yawe1);
 yawD_Error = (yawError-old_yaw_Error)/dt;
 
kYaw =-1; 
kD_Yaw =-0.8;

 % compute the yaw commands
 yawCmd = kYaw*yawError+kD_Yaw*yawD_Error;
 
 if abs(yawCmd) > 3.4 
 yawCmd = sign(yawCmd)*3.4; 
 end 


 

 errorArray = [yawError]; 


  if inc>1 
      if Time(inc,1)-Time(inc-1,1)>eps
          commandArray_Idf= [ux; uy; yawCmd; uz];
          FLAG(inc)=1;
  else
      commandArray_Idf= [CI(1,inc-1); CI(2,inc-1); CI(3,inc-1); CI(4,inc-1)];
      FLAG(inc)=0;
      end
  else
      commandArray_Idf= [ux; uy; yawCmd; uz];
       FLAG(inc)=1;
  end


move(p, 0.1, 'RotationSpeed', commandArray_Idf(3),'VerticalSpeed', commandArray_Idf(4),'roll', commandArray_Idf(2), 'pitch', commandArray_Idf(1));







end


