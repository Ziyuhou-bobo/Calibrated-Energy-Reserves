graphics_toolkit('qt'); P = pal();
lab = {'SAC','TD3','PPO','DSAC-N','DSAC-CVaR','CREST-C'};
cn = {'nominal','wind','sensor','traffic','energy','compound'};
A = dlmread('../pa/data/fig_main_std.txt');
f = figure('visible','off','Units','centimeters','Position',[1 1 18 5.6],'Color','w');
% (a) distribution of the minimum air clearance under the traffic shift
ax = subplot(1,3,1); hold on; hh = []; mm = {'sac','dsacn','crest'}; ci = [1 4 6];
for i = 1:3
  fn = ['../pa/data/ecdf_' mm{i} '.txt'];
  if exist(fn,'file') ~= 2, continue; end
  E = dlmread(fn);
  hh(end+1) = plot(E(:,1), 100*E(:,2), '-', 'Color', P.col(ci(i),:), 'LineWidth', 1.2);
end
plot([0 0],[0 100],'--','Color',[0.3 0.3 0.3],'LineWidth',0.7);
xlim([-5 60]); ylim([0 40]); xlabel('minimum air clearance (m)'); ylabel('missions below (%)');
legend(hh, {'SAC','DSAC-N','CREST-C'}, 'Location','southeast','FontSize',6); legend boxoff;
title('(a) separation, traffic shift','FontWeight','normal'); mstyle(ax);
% (b) composition of collisions
ax = subplot(1,3,2); hold on;
Ya = zeros(6,6); Yb = zeros(6,6);
for m = 1:6, for c = 1:6, r = (m-1)*6 + c; Ya(c,m) = A(r,3); Yb(c,m) = A(r,4); end; end
h = groupbar(Ya, P.col); [G,S] = size(Ya); w = 0.8/S;
for i = 1:S
  x = (1:G) + (i - (S+1)/2)*w;
  bar(x, Ya(:,i) + Yb(:,i), w*0.92, 'FaceColor', 'none', 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 0.35);
end
set(gca,'XTickLabel',cn,'XTickLabelRotation',30); ylabel('collisions (%)');
title('(b) aircraft (filled) and building (outline)','FontWeight','normal'); mstyle(ax); set(gca,'XGrid','off');
% (c) efficiency: energy against delivery time
ax = subplot(1,3,3); hold on; hh = [];
for m = 1:6
  x = zeros(1,6); y = zeros(1,6);
  for c = 1:6, r = (m-1)*6 + c; x(c) = A(r,9); y(c) = A(r,8); end
  hh(m) = plot(x, y, P.mk{m}, 'Color', P.col(m,:), 'MarkerFaceColor', P.col(m,:), 'MarkerSize', 4, 'LineStyle','none');
end
xlabel('delivery time (s)'); ylabel('energy per mission (kJ)');
legend(hh, lab, 'Location','northwest','FontSize',5.4); legend boxoff;
title('(c) efficiency across conditions','FontWeight','normal'); mstyle(ax);
savefig2A(f,'fig_extra',18,5.6); disp('done');
