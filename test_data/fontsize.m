function fontsize(size,h)
if nargin==1
   h=gca;
end
set(gca,'FontSize',size,'FontName','Times')
xlabel(get(get(gca,'xlabel'),'String'),'FontSize',size,'FontWeight','bold','FontAngle','normal','FontName','Helvetica') 
ylabel(get(get(gca,'ylabel'),'String'),'FontSize',size,'FontWeight','bold','FontAngle','normal','FontName','Helvetica') 
