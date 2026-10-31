graphics_toolkit('qt'); P = pal();
A = dlmread('../pa/data/fig_pareto.txt');   % rows: policy x rule(5 fixed, 5 online) x cond(nominal,wind,energy)
cn = {'nominal','wind shift','energy shift'};
mg = {'1.0','1.2','1.5','2.0','2.5'}; dl = {'0.02','0.05','0.10','0.20','0.30'};
f = figure('visible','off','Units','centimeters','Position',[1 1 18 5.8],'Color','w');
for c = 1:3
  ax = subplot(1,3,c); hold on;
  xf = zeros(1,5); yf = zeros(1,5); xo = zeros(1,5); yo = zeros(1,5);
  for i = 1:5
    r = (i-1)*3 + c;            xf(i) = A(r,1); yf(i) = A(r,2);
    r = (5+i-1)*3 + c;          xo(i) = A(r,1); yo(i) = A(r,2);
  end
  h1 = plot(xf, yf, '-o', 'Color', P.gcol(1,:), 'MarkerFaceColor', P.gcol(1,:), 'MarkerSize', 3.5, 'LineWidth', 1.1);
  h2 = plot(xo, yo, '-s', 'Color', P.gcol(4,:), 'MarkerFaceColor', P.gcol(4,:), 'MarkerSize', 3.5, 'LineWidth', 1.3);
  ymax = max([yf yo]) * 1.35 + 1e-3; ylim([-0.04*ymax ymax]); dy = 0.045 * ymax;
  for i = 1:5
    text(xf(i), yf(i)+dy, mg{i}, 'FontSize', 5.4, 'Color', P.gcol(1,:), 'HorizontalAlignment','center','FontName','Helvetica');
    text(xo(i), yo(i)-dy, dl{i}, 'FontSize', 5.4, 'Color', P.gcol(4,:), 'HorizontalAlignment','center','FontName','Helvetica');
  end
  xlabel('deliveries abandoned (%)'); if c == 1, ylabel('energy depletion (%)'); end
  title(['(' char(96+c) ') ' cn{c}], 'FontWeight','normal');
  if c == 1, legend([h1 h2], {'fixed margin','online conformal'}, 'Location','northeast','FontSize',6); legend boxoff; end
  mstyle(ax);
end
savefig2A(f,'fig_pareto',18,5.8); disp('done');
