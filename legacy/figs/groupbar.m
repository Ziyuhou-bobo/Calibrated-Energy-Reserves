function h = groupbar(Y, C, E)
  % grouped bar chart: Y is G x S (groups x series), C is S x 3 colours, E optional G x S x 2 CI bounds
  [G, S] = size(Y); w = 0.8 / S; h = zeros(1, S); hold on;
  for i = 1:S
    x = (1:G) + (i - (S + 1) / 2) * w;
    h(i) = bar(x, Y(:, i), w * 0.92, 'FaceColor', C(i, :), 'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.35);
    if nargin > 2 && ~isempty(E)
      for g = 1:G
        if isfinite(E(g, i, 1))
          plot([x(g) x(g)], [E(g, i, 1) E(g, i, 2)], '-', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.5);
          plot(x(g) + [-1 1] * w * 0.22, [E(g, i, 1) E(g, i, 1)], '-', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.5);
          plot(x(g) + [-1 1] * w * 0.22, [E(g, i, 2) E(g, i, 2)], '-', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.5);
        end
      end
    end
  end
  xlim([0.4 G + 0.6]); set(gca, 'XTick', 1:G);
end
