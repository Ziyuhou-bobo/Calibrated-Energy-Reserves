function P = pal()
  % study-specific palette (methods and guards)
  P.names = {'SAC', 'TD3', 'PPO', 'DSAC-N', 'DSAC-CVaR', 'CREST'};
  P.keys  = {'sac', 'td3', 'ppo', 'dsacn', 'dsaccvar', 'crest'};
  P.col = [0.35 0.44 0.66; 0.50 0.56 0.22; 0.58 0.38 0.58; 0.82 0.60 0.20; 0.16 0.52 0.55; 0.78 0.16 0.14];
  P.mk = {'s', 'd', '^', 'v', 'o', 'p'};
  P.ls = {'-', '-', '-', '-', '-', '-'};
  P.gnames = {'1.2\times fixed', '1.5\times fixed', 'split conformal', 'online conformal'};
  P.gkeys = {'std', 'std15', 'static', 'online'};
  P.gcol = [0.50 0.50 0.50; 0.66 0.52 0.36; 0.24 0.40 0.64; 0.78 0.16 0.14];
end
