graphics_toolkit('qt'); P = pal();
lab = {'SAC','TD3','PPO','DSAC-N','DSAC-CVaR','CREST-C'};
cn = {'nominal','wind','sensor','traffic','energy','compound'};
f = figure('visible','off','Units','centimeters','Position',[1 1 18 10.4],'Color','w');
for pnl = 1:2
  g = {'std','online'}{pnl};
  A = dlmread(['../pa/data/fig_main_' g '.txt']);   % rows: method-major, condition-minor
  Y = zeros(6,6); E = zeros(6,6,2);
  for m = 1:6
    for c = 1:6
      r = (m-1)*6 + c; Y(c,m) = A(r,1); E(c,m,1) = A(r,14); E(c,m,2) = A(r,15);
    end
  end
  ax = subplot(2,1,pnl); h = groupbar(Y, P.col, E);
  set(gca,'XTickLabel',cn); ylabel('catastrophic loss rate (%)');
  ylim([0 max(62, 1.25*max(Y(:)))]);
  ttl = {'(a) fixed 1.2\times return-to-home reserve','(b) online conformal reserve'}{pnl};
  title(ttl,'FontWeight','normal');
  if pnl == 1, legend(h, lab, 'Location','northwest','Orientation','horizontal','FontSize',6); legend boxoff; end
  mstyle(ax); set(gca,'XGrid','off');
end
savefig2A(f,'fig_main',18,10.4); disp('done');
