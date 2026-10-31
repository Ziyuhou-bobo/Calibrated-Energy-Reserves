graphics_toolkit('qt'); P = pal();
A = dlmread('../pa/data/fig_learn.txt');   % rows: method x {succ,coll,ret} x {mean,sd}
grid_s = 250:250:25000; nm = 7; keys = {'sac','td3','ppo','dsacn','dsaccvar','crest','crestret'};
lab = {'SAC','TD3','PPO','DSAC-N','DSAC-CVaR','CREST-C','CREST-R'};
col = [P.col; 0.45 0.30 0.62];
f = figure('visible','off','Units','centimeters','Position',[1 1 18 5.4],'Color','w');
ttl = {'(a) delivery success','(b) collisions','(c) mean mission return'};
yl = {'training success rate','training collision rate','mean return'};
for pnl = 1:3
  ax = subplot(1,3,pnl); hold on; hh = [];
  for m = 1:nm
    r = (m-1)*6 + (pnl-1)*2 + 1; mu = A(r,:); sd = A(r+1,:);
    k = isfinite(mu); x = grid_s(k); y = mu(k); s = sd(k);
    if sum(k) < 2, continue; end
    patch([x fliplr(x)]/1000, [y-s fliplr(y+s)], col(m,:), 'FaceAlpha', 0.12, 'EdgeColor', 'none');
    hh(end+1) = plot(x/1000, y, '-', 'Color', col(m,:), 'LineWidth', 1.2);
  end
  for m = 1:nm
    r = (m-1)*6 + (pnl-1)*2 + 1; mu = A(r,:); k = isfinite(mu);
    if sum(k) < 2, continue; end
  end
  xlabel('environment steps (10^3)'); ylabel(yl{pnl}); title(ttl{pnl}, 'FontWeight','normal'); xlim([0 25]);
  if pnl == 1, legend(hh, lab(1:numel(hh)), 'Location','southeast','FontSize',5.6); legend boxoff; end
  mstyle(ax);
end
savefig2A(f, 'fig_learn', 18, 5.4); disp('done');
