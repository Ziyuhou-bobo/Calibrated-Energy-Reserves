graphics_toolkit('qt'); P = pal();
f = figure('visible', 'off', 'Units', 'centimeters', 'Position', [1 1 18 8.4], 'Color', 'w');
ax = axes('Position', [0 0 1 1]); hold on; axis off; xlim([0 18]); ylim([0 8.4]);
cEnv = [0.93 0.95 0.97]; cRisk = [0.99 0.93 0.90]; cLearn = [0.92 0.96 0.95]; cGuard = [0.96 0.94 0.90];
eEnv = [0.35 0.44 0.66]; eRisk = P.col(6, :); eLearn = P.col(5, :); eGuard = [0.66 0.52 0.36];
% environment
sbox(0.25, 1.2, 4.1, 6.0, '', cEnv, eEnv);
text(2.3, 6.75, 'Stochastic urban district', 'HorizontalAlignment', 'center', 'FontName', 'Helvetica', 'FontSize', 7.5, 'FontWeight', 'bold');
text(0.5, 4.55, {'\bullet mean wind, OU turbulence,', '   1-cosine gusts', '\bullet GNSS drift and multipath', '\bullet fleet traffic and', '   non-cooperative intruders', '\bullet battery, payload and', '   in-flight power faults', '\bullet shift families at test time:', '   wind, sensor, traffic, energy', '\bullet paired test districts', '   (common random numbers)'}, ...
     'FontName', 'Helvetica', 'FontSize', 6.6, 'VerticalAlignment', 'middle');
% hazard cues and schedule
sbox(5.3, 5.75, 3.6, 1.35, {'Hazard cues', 'h_t = (h_c, h_e, h_d, h_w)'}, cRisk, eRisk, 7);
sbox(9.8, 5.75, 4.3, 1.35, {'Hazard-adaptive risk level', '(HA-R and HA-C)', '\alpha_t = \alpha_{min} + (\alpha_b - \alpha_{min})(1 - H(h_t))'}, cRisk, eRisk, 7);
% learning block
sbox(9.8, 3.35, 4.3, 1.5, {'Policy under test \pi(a | o, \alpha)', 'SAC, TD3, PPO, DSAC-N,', 'DSAC-CVaR, HA-R, HA-C'}, cLearn, eLearn, 7);
sbox(15.05, 3.35, 2.7, 3.75, {'Quantile critics', 'return Z(o, a; \alpha)', 'HA-C adds', 'catastrophe cost', 'C(o, a; \alpha)'}, cLearn, eLearn, 7);
% conformal reserve block
sbox(5.3, 0.35, 5.3, 1.9, {'Energy reserve under test', 'fixed 1.2\times / 1.5\times, split conformal,', 'online conformal (\kappa_t, q_t, fleet calibrator \lambda_j)'}, cGuard, eGuard, 7);
sbox(11.5, 0.35, 6.25, 1.9, {'Return-to-home trigger', 'E_t < B_t + E_{drop},  B_t = reserve on E_{nom}(w^{est}_t)', 'energy-optimal return airspeed'}, cGuard, eGuard, 7);
% arrows
sarrow([4.35 6.425; 5.3 6.425], eRisk);
sarrow([8.9 6.425; 9.8 6.425], eRisk);
sarrow([11.95 5.75; 11.95 4.85], eRisk);
text(12.1, 5.3, '\alpha_t', 'FontName', 'Helvetica', 'FontSize', 7, 'Color', eRisk);
sarrow([14.1 6.425; 16.4 6.425; 16.4 7.1 - 0.001], eRisk, 0.8);
sarrow([4.35 4.1; 9.8 4.1], eLearn); text(6.4, 4.3, 'observation o_t', 'FontName', 'Helvetica', 'FontSize', 6.8, 'Color', eLearn);
sarrow([15.05 4.1; 14.1 4.1], eLearn); text(14.575, 4.45, 'tail', 'HorizontalAlignment', 'center', 'FontName', 'Helvetica', 'FontSize', 6.2, 'Color', eLearn);
sarrow([11.95 3.35; 11.95 2.8; 4.35 2.8], eLearn); text(6.4, 3.0, 'action a_t (goal-frame velocity)', 'FontName', 'Helvetica', 'FontSize', 6.8, 'Color', eLearn);
sarrow([4.35 1.3; 5.3 1.3], eGuard); text(4.42, 1.02, 'telemetry', 'FontName', 'Helvetica', 'FontSize', 6.2, 'Color', eGuard);
sarrow([10.6 1.3; 11.5 1.3], eGuard);
sarrow([14.6 0.35; 14.6 0.15; 2.3 0.15; 2.3 1.2], eGuard);
text(7.2, 0.0 + 0.02, '', 'FontSize', 1);
text(15.1, 0.02, 'switch goal to nearest pad', 'FontName', 'Helvetica', 'FontSize', 6.2, 'Color', eGuard, 'VerticalAlignment', 'bottom');
sarrow([2.3 7.2; 2.3 7.8; 16.4 7.8; 16.4 7.12], [0.45 0.45 0.45], 0.8);
text(6.8, 8.0, 'replay stores hazard cues; risk level relabelled at every update (HA-R, HA-C)', 'FontName', 'Helvetica', 'FontSize', 6.5, 'Color', [0.35 0.35 0.35]);
savefig2A(f, 'fig_schematic', 18, 8.4);
disp('done');
