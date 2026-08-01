(* ::Package:: *)

(* TDSE for SSE_third_test.ipynb
   i dC/dt = H(t) C(t),  H(t) = H0 - F_x(t) mu_x - F_y(t) mu_y - F_z(t) mu_z

   Exact solution of the ODE system via NDSolve.
*)

ClearAll["Global`*"];

hartreeToEv = 27.2114;
auToAs = 24.18884326505;

(* --- Pulse parameters (SSE_third_test.ipynb) --- *)
intensity = 1.*^14;
fwhm = 1000.;
omegaXEv = 277.;
omegaOEv = 8.13947;
timeSpan = 300.;
pulseWindow = {0, timeSpan};
outputDt = 5.*^-3;  (* only used to sample the NDSolve solution for plots/export *)

(* --- Energies (ci_energy.inp, eV -> Hartree) --- *)
energyEv = {0., 8.13947, 277.24949};
energies = energyEv/hartreeToEv;
H0 = DiagonalMatrix[energies];

(* --- Dipoles (ci_mut.inp) --- *)
muX = {
  {0., 0.09440938, 0.00775143},
  {0.09440938, 0., -0.00010397},
  {0.00775143, -0.00010397, 0.}
};
muY = {
  {0., 0.02448033, -0.00503278},
  {0.02448033, 0., -0.00257584},
  {-0.00503278, -0.00257584, 0.}
};
muZ = {
  {0., 0.06490973, -0.01828601},
  {0.06490973, 0., 0.00042899},
  {-0.01828601, 0.00042899, 0.}
};

(* --- Pulses (ElectricFieldPulse, gaussian_sin) --- *)
eMax = Sqrt[intensity/3.51*^16];
sigma = (fwhm/auToAs)/(2. Sqrt[2. Log[2.]]);
t0 = Ceiling[5. sigma];
omegaX = omegaXEv/hartreeToEv;
omegaO = omegaOEv/hartreeToEv;

gaussianSinPulse[omega_, t_] :=
  eMax Exp[-(t - t0)^2/(2. sigma^2)] Sin[omega t];

fieldX[t_] := gaussianSinPulse[omegaO, t];  (* optical on x *)
fieldY[t_] := gaussianSinPulse[omegaX, t];  (* X-ray on y *)
fieldZ[t_] := 0.;

interactionH[t_] := -(
   fieldX[t] muX +
   fieldY[t] muY +
   fieldZ[t] muZ
   );

H[t_] := H0 + interactionH[t];

cInitial = {1., 0., 0.};

(* --- FFT of pulses (same convention as ElectricFieldPulse.fft_pulse) --- *)

fftShift[data_] := RotateLeft[data, Floor[Length[data]/2]];

fftFreqAu[n_Integer?Positive, dt_?Positive] := Module[
  {nh = Floor[(n - 1)/2] + 1, val = 1/(n dt)},
  Join[
    Range[0, nh - 1],
    -Range[n - nh, 1, -1]
  ] * val
];

nPulseSteps = Round[timeSpan/outputDt];
pulseTimeGrid = N@Subdivide[0., timeSpan, nPulseSteps - 1];

pulseSpectrum[pulseFn_] := Module[{samples, n, spec, freqsHz},
  samples = N@Table[pulseFn[t], {t, pulseTimeGrid}];
  n = Length[samples];
  If[n < 2,
    Return[<|"freqEv" -> {}, "magnitude" -> {}|>]
  ];
  spec = fftShift@Fourier[samples, FourierParameters -> {-1, 1}];
  freqsHz = fftShift@fftFreqAu[n, outputDt];
  <|
    "freqEv" -> N[2. Pi freqsHz hartreeToEv],
    "magnitude" -> N[Abs[spec]]
  |>
];

spectrumFwhm[x_List, y_List] := Module[
  {halfMax = Max[y]/2., idx, i1, i2, x1, x2, y1, y2},
  idx = Flatten@Position[y, _?(# >= halfMax &), {1}, 1];
  If[idx === {} || idx[[1]] <= 1 || idx[[1]] >= Length[y], Return[Indeterminate]];
  i1 = idx[[1]] - 1;
  i2 = idx[[1]];
  {x1, y1} = {x[[i1]], y[[i1]]};
  {x2, y2} = {x[[i2]], y[[i2]]};
  If[y2 === y1, Return[x2 - x1]];
  x1 + (halfMax - y1) (x2 - x1)/(y2 - y1)
];

estimateCentralFrequency[freqs_List, mags_List] :=
  freqs[[First@Ordering[mags, -1]]];

positiveSpectrumPairs[spectrum_Association] :=
  Select[
    Transpose[{spectrum["freqEv"], spectrum["magnitude"]}],
    Positive[#[[1]]] &
  ];

windowedSpectrum[spectrum_Association, limits_Association] := Module[
  {fmin = limits["freq"][[1]], fmax = limits["freq"][[2]]},
  Select[
    positiveSpectrumPairs[spectrum],
    fmin <= #[[1]] <= fmax &
  ]
];

pulsePlotLimits[spectrum_Association, carrierEv_] := Module[
  {pairs, freqs, mags, central, fwhm, freqWindow, timeWindow = 5. sigma},
  pairs = positiveSpectrumPairs[spectrum];
  freqs = pairs[[All, 1]];
  mags = pairs[[All, 2]];
  central = estimateCentralFrequency[freqs, mags];
  fwhm = spectrumFwhm[freqs, mags];
  central = If[NumericQ[central], central, carrierEv];
  freqWindow = If[NumericQ[fwhm], 2. fwhm, 20.];
  <|
    "time" -> {t0 - timeWindow, t0 + timeWindow},
    "freq" -> If[central - freqWindow < 0.,
      {0., central + freqWindow},
      {central - freqWindow, central + freqWindow}
    ]
  |>
];

spectrumLinePlot[spectrum_Association, limits_Association, label_] := Module[
  {data = windowedSpectrum[spectrum, limits], ymax},
  ymax = Max[data[[All, 2]]];
  If[! NumericQ[ymax] || ymax <= 0., ymax = 1.];
  ListLinePlot[data,
    PlotRange -> {limits["freq"], {0., 1.05 ymax}},
    AspectRatio -> 1/GoldenRatio,
    AxesLabel -> {"Energy (eV)", "Magnitude"},
    PlotLabel -> label,
    ImageSize -> 480,
    PlotRangePadding -> {{0.02, 0.02}, {0., 0.05}}
  ]
];

spectrumOptical = pulseSpectrum[fieldX];
spectrumXray = pulseSpectrum[fieldY];
limitsOptical = pulsePlotLimits[spectrumOptical, omegaOEv];
limitsXray = pulsePlotLimits[spectrumXray, omegaXEv];

fieldTimeSpectrumPlot[
  pulseFn_,
  spectrum_Association,
  limits_Association,
  timeLabel_,
  spectrumLabel_
] := GraphicsRow[
  {
    Plot[
      pulseFn[t],
      {t, Sequence @@ limits["time"]},
      PlotRange -> {limits["time"], {-eMax, eMax}},
      AspectRatio -> 1/GoldenRatio,
      AxesLabel -> {"t (a.u.)", "E (a.u.)"},
      PlotLabel -> timeLabel,
      ImageSize -> 480
    ],
    spectrumLinePlot[spectrum, limits, spectrumLabel]
  },
  ImageSize -> 980
];

(* ============================================================
   Exact ODE solution: i c'[t] = H[t] . c[t]
   ============================================================ *)

Print["Solving i c'[t] = H[t] . c[t] with NDSolve ..."];

odeSolution = NDSolveValue[
  {
    I c1'[t] == H[t][[1, 1]] c1[t] + H[t][[1, 2]] c2[t] + H[t][[1, 3]] c3[t],
    I c2'[t] == H[t][[2, 1]] c1[t] + H[t][[2, 2]] c2[t] + H[t][[2, 3]] c3[t],
    I c3'[t] == H[t][[3, 1]] c1[t] + H[t][[3, 2]] c2[t] + H[t][[3, 3]] c3[t],
    c1[0] == cInitial[[1]],
    c2[0] == cInitial[[2]],
    c3[0] == cInitial[[3]]
  },
  {c1, c2, c3},
  {t, 0., timeSpan},
  AccuracyGoal -> 12,
  PrecisionGoal -> 12
];

cAt[t_?NumericQ] := {odeSolution[[1]][t], odeSolution[[2]][t], odeSolution[[3]][t]};
populationAt[t_?NumericQ] := Abs[cAt[t]]^2;

timeGrid = Range[0., timeSpan, outputDt];
plotStride = Max[1, Round[(Length[timeGrid] - 1)/5000]];
plotTimes = timeGrid[[1 ;; -1 ;; plotStride]];

populationGrid = Table[populationAt[t], {t, timeGrid}];
totalPopulation = Total[populationGrid, {2}];

subsample[data_] := data[[1 ;; -1 ;; plotStride]];

idxCheck = Round[90./outputDt] + 1;
Print["Population at t = 90 a.u.: ", populationGrid[[idxCheck]]];
Print["Norm at t = 90 a.u.: ", Total[populationGrid[[idxCheck]]]];

(* ============================================================
   Plots
   ============================================================ *)

Print["Generating plots ..."];

plotFieldOptical = fieldTimeSpectrumPlot[
  fieldX,
  spectrumOptical,
  limitsOptical,
  "Optical field (F_x)",
  "FFT of optical pulse"
];

plotFieldXray = fieldTimeSpectrumPlot[
  fieldY,
  spectrumXray,
  limitsXray,
  "X-ray field (F_y)",
  "FFT of X-ray pulse"
];

plotPopulations = ListLinePlot[
  Table[
    Transpose[{plotTimes, subsample[populationGrid][[All, i]]}],
    {i, 1, 3}
  ],
  PlotRange -> {pulseWindow, Automatic},
  PlotLegends -> {"|C0(t)|^2", "|C1(t)|^2", "|C2(t)|^2"},
  AxesLabel -> {"t (a.u.)", "Population"},
  PlotLabel -> "Populations (NDSolve)",
  ImageSize -> 640
];

statePlot[stateIndex_, label_] := ListLinePlot[
  Transpose[{plotTimes, subsample[populationGrid][[All, stateIndex]]}],
  PlotRange -> {pulseWindow, Automatic},
  AxesLabel -> {"t (a.u.)", "Population"},
  PlotLabel -> label,
  ImageSize -> 480
];

plotPopulationGrid = GraphicsGrid[
  {
    {statePlot[1, "State 0"], statePlot[2, "State 1"]},
    {statePlot[3, "State 2"],
     ListLinePlot[
       Transpose[{plotTimes, subsample[totalPopulation]}],
       PlotRange -> {pulseWindow, {0.999, 1.001}},
       AxesLabel -> {"t (a.u.)", "Population"},
       PlotLabel -> "Total population",
       ImageSize -> 480
     ]}
  },
  ImageSize -> 1000
];

plotFieldOptical
plotFieldXray
plotPopulations
plotPopulationGrid

exportPath = FileNameJoin[{Directory[], "SSE_third_test_mathematica_populations.csv"}];
Export[
  exportPath,
  MapThread[Prepend[#2, #1] &, {plotTimes, subsample[populationGrid]}],
  "CSV"
];
Print["Exported populations to: ", exportPath];

$results = <|
  "odeSolution" -> odeSolution,
  "populationGrid" -> populationGrid,
  "timeGrid" -> timeGrid,
  "populationAt90" -> populationGrid[[idxCheck]]
|>;



Print
