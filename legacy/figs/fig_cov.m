graphics_toolkit('qt'); P = pal();
A = dlmread('../pa/data/fig_cov.txt');   % rows: policy(crest,sac) x guard(std,std15,static,online) x cond
cn = {'nominal','wind','sensor','traffic','energy','compound'};
f = figure('visible','off','Units','centimeters','Position',[1 1 18 5.6],'Color','w');
% (a) coverage of the return-energy reserve
ax = subplot(1,3,1);
Y = zeros(6,4);
for g = 1:4, for c = 1:6, Y(c,g) = A((g-1)*6 + c, 1); end; end
h = groupbar(Y, P.gcol); plot([0.4 6.6],[95 95],'--','Color',[0.2 0.2 0.2],'LineWidth',0.8);
set(gca,'XTickLabel',cn,'XTickLabelRotation',30); ylabel('return legs within reserve (%)'); ylim([0 105]);
legend(h, P.gnames, 'Location','southwest','FontSize',5.6); legend boxoff;
title('(a) reserve coverage','FontWeight','normal'); mstyle(ax); set(gca,'XGrid','off');
% (b) energy depletion
ax = subplot(1,3,2);
for g = 1:4, for c = 1:6, Y(c,g) = A((g-1)*6 + c, 3); end; end
groupbar(Y, P.gcol); set(gca,'XTickLabel',cn,'XTickLabelRotation',30); ylabel('energy depletion (%)');
title('(b) depletion losses','FontWeight','normal'); mstyle(ax); set(gca,'XGrid','off');
% (c) aborted deliveries
ax = subplot(1,3,3);
for g = 1:4, for c = 1:6, Y(c,g) = A((g-1)*6 + c, 4); end; end
groupbar(Y, P.gcol); set(gca,'XTickLabel',cn,'XTickLabelRotation',30); ylabel('deliveries abandoned (%)');
title('(c) operational cost of the reserve','FontWeight','normal'); mstyle(ax); set(gca,'XGrid','off');
savefig2A(f,'fig_cov',18,5.6); disp('done');
