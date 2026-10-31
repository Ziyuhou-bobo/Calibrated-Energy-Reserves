graphics_toolkit('qt'); P = pal();
A = dlmread('../pa/data/fig_dial.txt');   % rows: alpha(5) x cond(4); cols: coll, loss, succ, tdel, minsep, energy
al = [0.1 0.25 0.5 0.75 1.0]; cn = {'nominal','sensor shift','traffic shift','compound'};
cc = [0.35 0.44 0.66; 0.16 0.52 0.55; 0.82 0.60 0.20; 0.78 0.16 0.14];
f = figure('visible','off','Units','centimeters','Position',[1 1 18 5.4],'Color','w');
ttl = {'(a) collisions','(b) catastrophic loss','(c) delivery time'}; col = [1 2 4];
yl = {'collision rate (%)','catastrophic loss rate (%)','delivery time (s)'};
for pnl = 1:3
  ax = subplot(1,3,pnl); hold on; hh = [];
  for c = 1:4
    y = zeros(1,5);
    for i = 1:5, y(i) = A((i-1)*4 + c, col(pnl)); end
    hh(c) = plot(al, y, '-o', 'Color', cc(c,:), 'MarkerFaceColor', cc(c,:), 'MarkerSize', 3.4, 'LineWidth', 1.1);
  end
  xlabel('base risk level \alpha_b'); ylabel(yl{pnl}); xlim([0.05 1.05]);
  title(ttl{pnl}, 'FontWeight','normal');
  if pnl == 1, legend(hh, cn, 'Location','northeast','FontSize',5.8); legend boxoff; end
  mstyle(ax);
end
savefig2A(f,'fig_dial',18,5.4); disp('done');
