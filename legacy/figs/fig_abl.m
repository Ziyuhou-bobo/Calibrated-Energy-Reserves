graphics_toolkit('qt'); P = pal();
A = dlmread('../pa/data/fig_abl.txt');   % rows: 2 families x 4 variants x 6 conds, then 2 x 6 reference rows
cn = {'nominal','wind','sensor','traffic','energy','compound'};
vn = {'full schedule','fixed risk level','no energy cue','proximity only','return-spread driven'};
cv = [0.78 0.16 0.14; 0.50 0.50 0.50; 0.82 0.60 0.20; 0.16 0.52 0.55; 0.45 0.30 0.62];
f = figure('visible','off','Units','centimeters','Position',[1 1 18 9.0],'Color','w');
for fi = 1:2
  ax = subplot(2,1,fi); Y = zeros(6,5);
  base = 48 + (fi-1)*6;               % reference rows for the full schedule
  for c = 1:6
    Y(c,1) = A(base + c, 1);
    for v = 1:4, Y(c,v+1) = A((fi-1)*24 + (v-1)*6 + c, 1); end
  end
  h = groupbar(Y, cv); set(gca,'XTickLabel',cn); ylabel('catastrophic loss rate (%)');
  ttl = {'(a) CREST-C: tail risk on the catastrophe cost','(b) CREST-R: tail risk on the return'}{fi};
  title(ttl,'FontWeight','normal');
  if fi == 1, legend(h, vn, 'Location','northwest','Orientation','horizontal','FontSize',5.8); legend boxoff; end
  mstyle(ax); set(gca,'XGrid','off');
end
savefig2A(f,'fig_abl',18,9.0); disp('done');
