clc,clear

img=imread('pic1.png');
w=60;
aa=200;
bb=100;
img=img(aa:aa+w,bb:bb+w);
[N,M]=size(img);
sobel_y=[1, 0, -1;2, 0, -2;1, 0, -1];
sobel_x=[1, 2, 1;0, 0, 0;-1,-2,-1];
par_x=conv2(img,sobel_x,'same')
par_y=conv2(img,sobel_y,'same')
Vy=2*sum(sum(par_x.*par_y));
Vx=sum(sum(par_y.^2-par_x.^2));
theta=0.5*atan2(Vy,Vx);
imshow(img)
hold on
quiver(M/2,N/2,cos(theta),sin(theta))