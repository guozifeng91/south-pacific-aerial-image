(* ::Package:: *)

(* ::Title:: *)
(*Creating a HeatMap*)


(* ::Text:: *)
(*Execute ALL the code cells first to complete function definitions. (execute any cell and choose Yes when you are asked if should automatically evaluate all the initialization cells)*)
(**)
(*blue cells should be execute one by one, red cells are for visualization only, they can be delete but not recommended . white cells are for debug only.*)


(* ::Subsection:: *)
(*Getting the values from the model*)


(* ::Input:: *)
(*pPoints=Import[NotebookDirectory[]<>"eval_predict_paired.csv"]; (* import predicted paired *)*)
(*rPoints=Import[NotebookDirectory[]<>"eval_predict_rest.csv"]; (* import predicted rest *)*)
(**)
(*points=Join[pPoints,rPoints];*)
(*xy=points[[#]][[1;;2]]&/@Range@Length@points;*)


(* ::Subsection:: *)
(*Setting variables  *)


(* ::Input:: *)
(*w=17761; (* width, in pixels *)*)
(*h=25006; (* height, in pixels *)*)
(*step=400; *)
(*w1=Floor[w/step];*)
(*h1=Floor[h/step];*)
(*total=w1*h1;*)


(* ::Subsection:: *)
(*Gaussian Kernel*)


(* ::Input:: *)
(*pixels=Table[{i*step-step/2,j*step-step/2},{i,1,w1,1},{j,1,h1,1}];*)
(*pixels=Flatten[pixels,1];*)
(*Length@pixels*)


(* ::Input:: *)
(*(*normalize the coordinates*)*)
(*For[i=1,i<Length@pixels,i++,*)
(*pixels[[i]][[1]]/=w;*)
(*pixels[[i]][[2]]/=h;*)
(*]*)


(* ::Input:: *)
(*(*normalize the coordinates*)*)
(*For[i=1,i<Length@xy,i++,*)
(*xy[[i]][[1]]/=w;*)
(*xy[[i]][[2]]/=h;*)
(*]*)


GaussianKernel[pos_,points_, sig_]:=
Module[{dists},
dists=N@Exp[-sig*sig*SquaredEuclideanDistance[pos,#]]&/@points;
Return@Total@dists;
];


(* ::Input:: *)
(*sig=20;*)
(*kernelValues={};*)
(*Monitor[*)
(*For[i=1,i<=Length@pixels,i++,*)
(*AppendTo[kernelValues,GaussianKernel[pixels[[i]],xy,sig]];*)
(*],*)
(*{ProgressIndicator[i,{1,Length@pixels}],i}*)
(*];*)
(*minMax=MinMax[kernelValues];*)


(* ::Subsection:: *)
(*Getting the HeatMap  *)


(* ::Input:: *)
(*kernelValuesRescale=Chop@Rescale[kernelValues,minMax];*)
(*colors=ColorData["BlueGreenYellow"][#]&/@kernelValuesRescale;*)
(*img=Image[Transpose@Partition[colors,h1],ImageSize->Medium,ColorSpace->"RGB"]*)


xyRegular=points[[#]][[1;;2]]&/@Range@Length@points;
For[i=1,i<=Length@xyRegular,i++,
xyRegular[[i]][[2]]=h-xyRegular[[i]][[2]];
]



(* ::Input:: *)
(*gPoints=Graphics[{Black,PointSize[Tiny],Point[#]&/@xyRegular}];*)
(**)
(*heatMap=ImageResize[img,{w,h}];*)
(*g=Overlay[{img,gPoints}];*)
