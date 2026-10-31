function P = rotorP(v, m)
  % rotary-wing electrical power (W): Zeng, Xu and Zhang (2019) with momentum-theory weight scaling
  P0 = 79.86; PI0 = 88.63; UTIP = 120; V0H = 4.03; D0 = 0.6; RHO = 1.225; SOL = 0.05; AREA = 0.503; M0 = 2.0;
  w = m ./ M0; pind = PI0 .* w.^1.5; x = v.^2 ./ (2 .* V0H.^2 .* w);
  P = P0 .* (1 + 3 .* v.^2 ./ UTIP.^2) + pind .* sqrt(1 ./ (sqrt(1 + x.^2) + x)) + 0.5 .* D0 .* RHO .* SOL .* AREA .* v.^3;
end
