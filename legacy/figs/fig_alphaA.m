graphics_toolkit('qt'); P = pal();
A = dlmread('../pa/data/alpha_trace.txt');   % columns: h_c, h_e, h_d, h_w, alpha
t = 1:rows(A); cc = [0.78 0.16 0.14; 0.82 0.60 0.20; 0.16 0.52 0.55; 0.35 0.44 0.66];
f = figure('visible','off','Units','centimeters','Position',[1 1 12 7.6],'Color','w');
ax = subplot(2,1,1); hold on; hh = [];
for i = 1:4, hh(i) = plot(t, A(:,i), '-', 'Color', cc(i,:), 'LineWidth', 1.0); end
ylabel('hazard cue'); ylim([-0.03 1.03]); set(gca,'XTickLabel',{});
legend(hh, {'collision threat h_c','energy margin h_e','traffic density h_d','disturbance h_w'}, 'Location','northwest','Orientation','horizontal','FontSize',5.6); legend boxoff;
title('(a) hazard cues along one delivery', 'FontWeight','normal'); mstyle(ax);
ax = subplot(2,1,2); hold on;
plot(t, A(:,5), '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.2);
plot([1 max(t)], [0.1 0.1], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.7);
xlabel('mission time (s)'); ylabel('risk level \alpha_t'); ylim([0 0.8]);
title('(b) scheduled risk level', 'FontWeight','normal'); mstyle(ax);
savefig2A(f,'fig_alphaA',12,7.6); disp('done');
