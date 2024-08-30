function f=myfun(z,zdata)

theta=z(1)*pi/180;
x0=z(2);
y0=z(3);

len=length(zdata)/2;
xvec=zdata(1:len);
yvec=zdata(len+1:2*len);

Rmat=[cos(theta) sin(theta); -sin(theta) cos(theta)];

rotPos=Rmat*([xvec';yvec']-[x0;y0]);

f=[rotPos(1,:)';rotPos(2,:)'];