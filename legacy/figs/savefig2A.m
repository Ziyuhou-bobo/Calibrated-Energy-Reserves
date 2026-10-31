function savefig2A(f, name, w, h)
  % export vector EPS and PDF (plus a PNG preview) with exact paper size in centimetres
  set(f, 'PaperUnits', 'centimeters', 'PaperPosition', [0 0 w h], 'PaperSize', [w h]);
  print(f, '-depsc2', '-r600', ['outA/' name '.eps']);
  print(f, '-dpdf', '-r600', ['outA/' name '.pdf']);
  print(f, '-dpng', '-r200', ['outA/' name '.png']);
  close(f);
end
