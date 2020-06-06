function [fitresult, gof] = createFit(T1, SFR1, XCH41)

[xData, yData, zData] = prepareSurfaceData( T1, SFR1, XCH41 );

ft = 'linearinterp';

[fitresult, gof] = fit( [xData, yData], zData, ft, 'Normalize', 'on' );

plot( fitresult,'Style','Contour');

title('Colour Map')
xlabel('Monolayer 1')
ylabel('Monolayer 2')

colormap jet
grid on


