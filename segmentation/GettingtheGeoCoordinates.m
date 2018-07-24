(* ::Package:: *)

(* ::Title:: *)
(*Find Coordinates Class*)


(* ::Text:: *)
(*Execute ALL the code cells first to complete function definitions. (execute any cell and choose Yes when you are asked if should automatically evaluate all the initialization cells)*)
(**)
(*blue cells should be execute one by one, red cells are for visualization only, they can be delete but not recommended . white cells are for debug only.*)


(* ::Text:: *)
(*always note that the working coordinate system is graphic system NOT image system (flip the Y axis)*)


(* ::Subsubsection:: *)
(*Import*)


(* ::Input:: *)
(*pPoints=Import["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/05_Validation/eval_predict_paired.csv"];*)
(*rPoints=Import["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/05_Validation/eval_predict_rest.csv"];*)
(*boundingBox=Import["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/00_Data/boundingBox/boundingBox.shp","Data"];*)
(*boundingBoxG=Import["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/00_Data/boundingBox/boundingBox.shp"];*)
(*coordBB=Drop[ToExpression[StringDelete[StringDelete[ToString[boundingBox[[1]][[2]][[2]][[1]]],"Polygon[GeoPosition["],"]]"]],-1];*)


(* ::Subsubsection:: *)
(*Variables*)


(* ::Input:: *)
(*(*the range of geo-coordinate*)*)
(*startLat=coordBB[[1,2]];*)
(*startLog=coordBB[[1,1]];*)
(**)
(*endLat=coordBB[[3,2]];*)
(*endLog=coordBB[[3,1]];*)
(**)
(*rangeLat=endLat-startLat;*)
(*rangeLog=endLog-startLog;*)
(**)
(*(*size of the image in pixel*)*)
(*imgWidth=17761;*)
(*imgHeight=25006;*)


(* ::Input:: *)
(*(*fruits*)*)
(*banana=1;*)
(*coco=2;*)
(*mango=3;*)
(*papaya=4;*)


(* ::Subsubsection:: *)
(*Sort Coordinates and Class*)


(* ::Input:: *)
(*typeOfTrees=4;*)
(*(*here by default,the csv locates at the same directory with the notebook*)*)
(*trees=Flatten[{pPoints,rPoints},1];*)
(*(*transpose to select the 3rd channel,which represents the type of trees*)*)
(*treesT=Transpose@trees;*)
(*(*get the position (index) of each type of trees,note that the type starts from 0 not 1*)*)
(*indexOfEachType=Flatten@Position[treesT[[3]],#]&/@(Range[typeOfTrees]-1);*)
(*(*trees by types:a 2d array of[which type,which tree],the 3rd channel of trees are discarded*)*)
(*treesByType=trees[[#]][[1;;2]]&/@#&/@indexOfEachType;*)
(*(*all trees,in case of need*)*)
(*allTrees=Transpose@treesT[[1;;2]];*)


(* ::Subsubsection:: *)
(*GeoLocation of trees*)


coordClass [fruit_]:=
Block[{treeOfInterest,x,y,pointLat,pointLog,treeOfInterestGeoCoord,map},
treeOfInterest=treesByType[[fruit]];
x=treeOfInterest[[All,1]];
y=treeOfInterest[[All,2]];
(*here are the coords*)
pointLat=((#/imgWidth)*rangeLat+startLat)&/@x;
pointLog=((#/imgHeight)*rangeLog+startLog)&/@y;
treeOfInterestGeoCoord=Transpose[{pointLog,pointLat}];
map=GeoGraphics[{{Gray,PointSize[Small],Point[GeoPosition[treeOfInterestGeoCoord]]},{White,Thin,boundingBox[[1]][[2]][[2]]}}(*,GeoBackground->"Satellite",GeoServer\[Rule]"DigitalGlobe"*)];
<|
"coord"-> treeOfInterestGeoCoord,
"map"-> map
|>
]


(* ::Input:: *)
(*bananaCoords=coordClass[banana];*)
(*bananaCoords["map"]*)
(**)
(*Export["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/04_Submition/COORDS/bananaCoordinates.csv",bananaCoords["coord"]];*)
(*Export["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/04_Submition/COORDS/bananaMap.pdf",bananaCoords["map"]];*)


(* ::Input:: *)
(*cocoCoords=coordClass[coco];*)
(*cocoCoords["map"]*)
(**)
(*Export["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/04_Submition/COORDS/cocoCoordinates.csv",cocoCoords["coord"]];*)
(*Export["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/04_Submition/COORDS/cocoMap.pdf",cocoCoords["map"]];*)


(* ::Input:: *)
(*mangoCoords=coordClass[mango];*)
(*mangoCoords["map"]*)
(**)
(*Export["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/04_Submition/COORDS/mangoCoordinates.csv",mangoCoords["coord"]];*)
(*Export["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/04_Submition/COORDS/mangoMap.pdf",mangoCoords["map"]];*)


(* ::Input:: *)
(*papayaCoords=coordClass[papaya];*)
(*papayaCoords["map"]*)
(**)
(*Export["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/04_Submition/COORDS/papayaCoordinates.csv",papayaCoords["coord"]];*)
(*Export["/Users/karlasaldana/polybox/Aerial Imagery of South Pacific Islands/04_Submition/COORDS/papayaMap.pdf",papayaCoords["map"]];*)
