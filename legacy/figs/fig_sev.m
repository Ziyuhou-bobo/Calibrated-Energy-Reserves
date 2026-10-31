graphics_toolkit('qt'); P = pal();
A = dlmread('../pa/data/fig_sev.txt');   % rows: family x severity(5) x method(6)
fam = {'wind','sensor','traffic','energy'}; sv = [0 0.25 0.5 0.75 1];
lab = {'SAC','TD3','PPO','DSAC-N','DSAC-CVaR','CREST-C'};
f = figure('visible','off','Units','centimeters','Position',[1 1 18 9.6],'Color','w');
for fi = 1:4
  ax = subplot(2,2,fi); hold on; hh = [];
  for m = 1:6
    y = zeros(1,5);
    for si = 1:5, y(si) = A((fi-1)*30 + (si-1)*6 + m, 1); end
    hh(m) = plot(sv, y, ['-' P.mk{m}], 'Color', P.col(m,:), 'MarkerFaceColor', P.col(m,:), 'MarkerSize', 3.4, 'LineWidth', 1.1);
  end
  xlabel('shift severity s'); ylabel('catastrophic loss rate (%)'); xlim([-0.03 1.03]);
  title(['(' char(96+fi) ') ' fam{fi} ' shift'], 'FontWeight','normal');
  if fi == 1, legend(hh, lab, 'Location','northwest','FontSize',5.8); legend boxoff; end
  mstyle(ax);
end
savefig2A(f,'fig_sev',18,9.6); disp('done');
