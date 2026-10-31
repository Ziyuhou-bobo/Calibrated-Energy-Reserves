function mstyle(ax, fs)
  % MATLAB-like axes styling: boxed axes, inward ticks, light major grid
  if nargin < 2, fs = 8; end
  set(ax, 'Box', 'on', 'TickDir', 'in', 'FontName', 'Helvetica', 'FontSize', fs, 'LineWidth', 0.6, ...
      'XGrid', 'on', 'YGrid', 'on', 'GridLineStyle', '-', 'GridColor', [0.15 0.15 0.15], 'GridAlpha', 0.15, ...
      'TickLength', [0.012 0.02], 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15]);
end
