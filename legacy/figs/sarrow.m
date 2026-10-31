function sarrow(xy, col, lw)
  % poly-line arrow through the points in xy (n x 2) with a filled head at the last point
  if nargin < 2, col = [0.2 0.2 0.2]; end
  if nargin < 3, lw = 1.0; end
  plot(xy(:, 1), xy(:, 2), '-', 'Color', col, 'LineWidth', lw);
  p1 = xy(end - 1, :); p2 = xy(end, :); d = p2 - p1; d = d / norm(d); n = [-d(2) d(1)];
  L = 0.22; W = 0.09;
  tri = [p2; p2 - L * d + W * n; p2 - L * d - W * n];
  patch(tri(:, 1), tri(:, 2), col, 'EdgeColor', col);
end
