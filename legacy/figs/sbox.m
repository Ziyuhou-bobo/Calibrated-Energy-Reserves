function sbox(x, y, w, h, txt, fc, ec, fs, fw)
  % rounded block for schematics: lower-left corner (x,y), width w, height h
  if nargin < 8, fs = 7; end
  if nargin < 9, fw = 'normal'; end
  rectangle('Position', [x y w h], 'Curvature', 0.08, 'FaceColor', fc, 'EdgeColor', ec, 'LineWidth', 0.9);
  text(x + w / 2, y + h / 2, txt, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
       'FontName', 'Helvetica', 'FontSize', fs, 'FontWeight', fw, 'Color', [0.1 0.1 0.1]);
end
