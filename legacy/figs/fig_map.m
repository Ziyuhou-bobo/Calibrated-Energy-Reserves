graphics_toolkit('qt'); P = pal();
OB = dlmread('../pa/data/map_obs.txt'); PD = dlmread('../pa/data/map_pads.txt'); GO = dlmread('../pa/data/map_goal.txt');
meth = {'sac','crest'}; tt = {'(a) SAC, fixed reserve','(b) CREST-C, conformal reserve'};
f = figure('visible','off','Units','centimeters','Position',[1 1 18 9.2],'Color','w');
uc = [0.35 0.44 0.66; 0.16 0.52 0.55; 0.82 0.60 0.20; 0.45 0.30 0.62; 0.20 0.50 0.25; 0.78 0.16 0.14];
for k = 1:2
  ax = subplot(1,2,k); hold on;
  X = dlmread(['../pa/data/map_pos_' meth{k} '.txt']); I = dlmread(['../pa/data/map_int_' meth{k} '.txt']);
  AC = dlmread(['../pa/data/map_act_' meth{k} '.txt']); OU = dlmread(['../pa/data/map_out_' meth{k} '.txt']);
  th = linspace(0, 2*pi, 40);
  for j = 1:rows(OB)
    patch(OB(j,1) + OB(j,3)*cos(th), OB(j,2) + OB(j,3)*sin(th), [0.82 0.82 0.84], 'EdgeColor', [0.55 0.55 0.58], 'LineWidth', 0.3);
  end
  for j = 1:columns(I)/2
    plot(I(:,2*j-1), I(:,2*j), ':', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.6);
  end
  N = columns(X)/2;
  for j = 1:N
    m = AC(:,j) > 0; x = X(m, 2*j-1); y = X(m, 2*j);
    plot(x, y, '-', 'Color', uc(j,:), 'LineWidth', 1.0);
    plot(x(1), y(1), 'o', 'Color', uc(j,:), 'MarkerFaceColor', 'w', 'MarkerSize', 3);
    if OU(j) == 3 || OU(j) == 4
      plot(x(end), y(end), 'x', 'Color', [0.85 0.1 0.1], 'MarkerSize', 7, 'LineWidth', 1.4);
    end
  end
  plot(PD(:,1), PD(:,2), 's', 'Color', [0.15 0.15 0.15], 'MarkerFaceColor', [0.35 0.35 0.35], 'MarkerSize', 5);
  plot(GO(:,1), GO(:,2), 'd', 'Color', [0.15 0.15 0.15], 'MarkerFaceColor', [0.95 0.85 0.35], 'MarkerSize', 4.5);
  axis equal; xlim([0 1600]); ylim([0 1600]); set(gca,'XTick',0:400:1600,'YTick',0:400:1600);
  xlabel('east (m)'); if k == 1, ylabel('north (m)'); end
  title(tt{k}, 'FontWeight','normal'); mstyle(ax);
end
savefig2A(f,'fig_map',18,9.2); disp('done');
