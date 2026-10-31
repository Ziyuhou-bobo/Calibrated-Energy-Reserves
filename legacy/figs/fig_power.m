graphics_toolkit('qt'); P = pal();
f = figure('visible', 'off', 'Units', 'centimeters', 'Position', [1 1 18 5.6], 'Color', 'w');
c4 = [0.16 0.52 0.55; 0.35 0.44 0.66; 0.82 0.60 0.20; 0.78 0.16 0.14];
% (a) power versus airspeed
ax = subplot(1, 3, 1); hold on;
v = linspace(0, 22, 300); pl = [0 0.6 1.2 1.8];
for k = 1:4, plot(v, rotorP(v, 2 + pl(k)), 'Color', c4(k, :), 'LineWidth', 1.3); end
xlabel('Airspeed v (m s^{-1})'); ylabel('Electrical power P(v,m) (W)'); xlim([0 22]);
legend(arrayfun(@(p) sprintf('payload %.1f kg', p), pl, 'UniformOutput', false), 'Location', 'north', 'FontSize', 6.5); legend boxoff;
title('(a)', 'FontWeight', 'normal'); mstyle(ax);
% (b) energy per ground kilometre versus airspeed under along-track wind
ax = subplot(1, 3, 2); hold on;
wv = [-8 -4 0 4 8]; cw = [0.16 0.52 0.55; 0.35 0.44 0.66; 0.2 0.2 0.2; 0.82 0.60 0.20; 0.78 0.16 0.14];
v = linspace(6, 22, 300); m = 3.0; VC = [8 10 12 14 16 18 20];
hb = zeros(1, numel(wv));
for k = 1:numel(wv)
  g = max(v - wv(k), 0.5); e = rotorP(v, m) .* 1000 ./ g / 1e3;
  hb(k) = plot(v, e, 'Color', cw(k, :), 'LineWidth', 1.3);
  gc = max(VC - wv(k), 0.5); ec = rotorP(VC, m) .* 1000 ./ gc / 1e3; [~, j] = min(ec);
  plot(VC(j), ec(j), 'o', 'MarkerSize', 4.5, 'MarkerFaceColor', cw(k, :), 'MarkerEdgeColor', 'k', 'LineWidth', 0.4);
end
ylim([5 45]); xlim([6 22]);
xlabel('Airspeed v (m s^{-1})'); ylabel('Energy per ground km (kJ)');
legend(hb, arrayfun(@(w) sprintf('headwind %+d m s^{-1}', w), wv, 'UniformOutput', false), 'Location', 'northeast', 'FontSize', 6.5); legend boxoff;
title('(b)', 'FontWeight', 'normal'); mstyle(ax);
% (c) nominal return energy for a 1 km leg: fixed cruise speed versus energy-optimal airspeed
ax = subplot(1, 3, 3); hold on;
w = linspace(-10, 10, 201); m = 2.0; PHI = 1.15; d = 1000;
Efix = rotorP(14, m) .* PHI .* d ./ max(14 - w, 1) / 1e3;
Eopt = zeros(size(w)); vopt = zeros(size(w));
for ii = 1:numel(w), gc = max(VC - w(ii), 1); ec = rotorP(VC, m) .* PHI .* d ./ gc; [Eopt(ii), jj] = min(ec); vopt(ii) = VC(jj); end
Eopt = Eopt / 1e3;
h1 = plot(w, Eopt, '-', 'Color', P.col(6, :), 'LineWidth', 1.5);
h3 = plot(w, Efix, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.3);
xlim([-10 10]); ylim([0 35]);
ylabel('Return energy for 1 km (kJ)'); xlabel('Headwind along return track (m s^{-1})');
title('(c)', 'FontWeight', 'normal'); mstyle(ax);
pos = get(ax, 'Position');
ax2 = axes('Position', pos, 'Color', 'none', 'YAxisLocation', 'right', 'XAxisLocation', 'top', 'XTickLabel', {}, 'XLim', [-10 10], 'YLim', [6 22]);
hold(ax2, 'on'); h2 = stairs(ax2, w, vopt, '-', 'Color', P.col(5, :), 'LineWidth', 0.9);
set(ax2, 'Box', 'off', 'TickDir', 'in', 'FontName', 'Helvetica', 'FontSize', 8, 'LineWidth', 0.6, 'YColor', P.col(5, :), 'XColor', [0.15 0.15 0.15], 'XTick', get(ax, 'XTick'), 'YTick', 6:4:22);
ylabel(ax2, 'Selected airspeed (m s^{-1})');
legend(ax, [h1 h3], {'energy-optimal airspeed', 'fixed 14 m s^{-1} cruise'}, 'Location', 'southeast', 'FontSize', 6.5); legend(ax, 'boxoff');
savefig2A(f, 'fig_power', 18, 5.6);
disp('done');
