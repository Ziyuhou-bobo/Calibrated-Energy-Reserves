graphics_toolkit('qt'); P = pal();
cc = [0.35 0.44 0.66; 0.82 0.60 0.20; 0.78 0.16 0.14]; nm = {'nominal','wind','energy'};
f = figure('visible','off','Units','centimeters','Position',[1 1 18 5.4],'Color','w');
ax = subplot(1,2,1); hold on; hh = [];
for i = 1:3
  fn = ['../pa/data/lam_' nm{i} '.csv'];
  if exist(fn,'file') ~= 2, continue; end
  A = dlmread(fn, ',', 1, 0);
  hh(end+1) = plot(A(:,1), A(:,3), '-', 'Color', cc(i,:), 'LineWidth', 1.1);
end
xlabel('completed return legs'); ylabel('fleet reserve calibrator \lambda_j');
title('(a) online calibration of the reserve','FontWeight','normal');
legend(hh, nm, 'Location','southeast','FontSize',6); legend boxoff; mstyle(ax);
ax = subplot(1,2,2); hold on;
for i = 1:3
  fn = ['../pa/data/lam_' nm{i} '.csv'];
  if exist(fn,'file') ~= 2, continue; end
  A = dlmread(fn, ',', 1, 0);
  plot(A(:,1), 100*A(:,4), '-', 'Color', cc(i,:), 'LineWidth', 1.1);
end
plot([0 3000],[5 5],'--','Color',[0.2 0.2 0.2],'LineWidth',0.8);
xlabel('completed return legs'); ylabel('running miscoverage (%)'); ylim([0 30]);
title('(b) realised versus target miscoverage','FontWeight','normal'); mstyle(ax);
savefig2A(f,'fig_lam',18,5.4); disp('done');
