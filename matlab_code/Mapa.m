clc
clear all
close all

X1=xlsread('Auto.xlsx','Auto','C2:C296836');

Y1=xlsread('Auto.xlsx','Auto','E2:E296836');

Z1=xlsread('Auto.xlsx','Auto','H2:H296836');

X2=xlsread('Auto.xlsx','Auto','C2:C296836');

Y2=xlsread('Auto.xlsx','Auto','E2:E296836');

Z2=xlsread('Auto.xlsx','Auto','K2:K296836');

subplot(2,1,1)

createFit(X1, Y1, Z1)

colorbar

 title('Color Map')

subplot(2,1,2)

createFit2(X2, Y2, Z2)

colorbar

 title('Color Map 2')
 
 saveas(figure(1),'Color_PlotERR3.png')