extensions [rnd cf shell]

__includes [
  "natural_env.nls"
  "agents.nls"
  "plotter.nls"
]

to setup
  ca
  if not file-exists? "patches.csv"
  [regenerate-world]

  import-world "patches.csv"
  setup-agents
  setup-patches-score
  setup-plot
  reset-ticks
end

to regenerate-world
  show (shell:exec "py" "generate_world.py" world_file (word world-width) (word world-height))
end

to go
  if(count turtles = 0)
  [stop]
  update-natural-world
  update-agents

  update-plot
  tick
end
@#$#@#$#@
GRAPHICS-WINDOW
227
10
835
619
-1
-1
6.0
1
10
1
1
1
0
0
0
1
0
99
0
99
1
1
1
ticks
30.0

BUTTON
13
16
76
49
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
86
17
149
50
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
879
115
1051
148
fire_spread
fire_spread
0.0
50
12.4
.1
1
%
HORIZONTAL

SLIDER
879
74
1051
107
fire_duration
fire_duration
1
10
5.0
1
1
NIL
HORIZONTAL

SLIDER
885
567
1072
600
forest_regrowth_duration
forest_regrowth_duration
1
500
250.0
1
1
NIL
HORIZONTAL

SLIDER
883
436
1055
469
crop_rot_duration
crop_rot_duration
1
500
112.0
1
1
NIL
HORIZONTAL

SLIDER
884
394
1057
427
crop_growth_duration
crop_growth_duration
1
500
300.0
1
1
NIL
HORIZONTAL

SLIDER
878
35
1050
68
natural_fire_chance
natural_fire_chance
0
.1
0.001
.001
1
NIL
HORIZONTAL

SLIDER
886
526
1068
559
forest_mature_duration
forest_mature_duration
0
500
500.0
1
1
NIL
HORIZONTAL

TEXTBOX
887
18
1037
36
Fire parameters
12
0.0
1

TEXTBOX
895
370
1045
388
Farmland parameters
12
0.0
1

TEXTBOX
895
502
1045
520
Forest parameters
12
0.0
1

SLIDER
880
257
1083
290
young_forest_flammabillity
young_forest_flammabillity
0
1
0.35
.01
1
NIL
HORIZONTAL

SLIDER
881
300
1066
333
old_forest_flammabillity
old_forest_flammabillity
0
1
0.15
.01
1
NIL
HORIZONTAL

SLIDER
880
174
1057
207
farmland_flammabillity
farmland_flammabillity
0
1
0.35
.01
1
NIL
HORIZONTAL

SLIDER
881
214
1053
247
crops_flammabillity
crops_flammabillity
0
1
0.6
.01
1
NIL
HORIZONTAL

BUTTON
14
87
141
120
regenerate world
regenerate-world
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

CHOOSER
14
130
152
175
world_file
world_file
"world_1" "world_2" "world_3"
0

TEXTBOX
15
63
165
81
World file generation
12
0.0
1

TEXTBOX
17
324
167
342
Forester parameters
12
0.0
1

BUTTON
75
460
130
493
+
add-forester
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
14
460
69
493
-
remove-forester
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
12
347
212
380
initial-amount-of-foresters
initial-amount-of-foresters
0
100
51.0
1
1
NIL
HORIZONTAL

MONITOR
12
400
151
449
amount of foresters
count foresters
0
1
12

SLIDER
10
223
182
256
movement-speed
movement-speed
0
100
1.0
1
1
NIL
HORIZONTAL

TEXTBOX
33
553
183
571
Farmers parameters\n
11
0.0
1

SLIDER
10
577
197
610
initial-amount-of-farmers
initial-amount-of-farmers
0
100
90.0
1
1
NIL
HORIZONTAL

MONITOR
11
623
121
668
Amout of farmers
count farmers
17
1
11

BUTTON
10
677
65
710
-
remove-farmer
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
73
677
128
710
+
add-farmer
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
1169
46
1341
79
river-score
river-score
0
1
0.5
0.01
1
NIL
HORIZONTAL

SLIDER
1170
102
1342
135
city-score
city-score
0
1
0.5
0.01
1
NIL
HORIZONTAL

SLIDER
9
803
247
836
probability-to-turn-into-wasteland
probability-to-turn-into-wasteland
0
100
20.0
1
1
NIL
HORIZONTAL

SLIDER
10
716
182
749
max-seeds
max-seeds
0
50
10.0
1
1
NIL
HORIZONTAL

PLOT
1138
290
1466
557
Ecology stats
NIL
NIL
0.0
10.0
0.0
10.0
true
true
"" ""
PENS
"old forest" 1.0 0 -15456499 true "" "plot count patches with [ptype = \"old forest\"]"
"young forest" 1.0 0 -13210332 true "" "plot count patches with [ptype = \"young forest\"]"
"farmland" 1.0 0 -8330359 true "" "plot count patches with [ptype = \"farmland\"]"
"wasteland" 1.0 0 -5207188 true "" "plot count patches with [ptype = \"wasteland\"]"
"city" 1.0 0 -7500403 true "" "plot count patches with [ptype = \"city\"]"

SLIDER
10
269
218
302
max-travel-distance
max-travel-distance
1
world-width * sqrt 2
5.0
1
1
patches
HORIZONTAL

PLOT
1138
573
1469
839
Occupancies
NIL
NIL
0.0
10.0
0.0
10.0
true
true
"" ""
PENS
"Forestry" 1.0 0 -14333415 true "" "plot forestry"
"Harvesting" 1.0 0 -13840069 true "" "plot harvesting"
"Planting" 1.0 0 -5509967 true "" "plot planting-seeds"
"Idle" 1.0 0 -9276814 true "" "plot idling"
"Replanting forest" 1.0 0 -13210332 true "" "plot planting-trees"
"Igniting" 1.0 0 -2674135 true "" "plot igniting"

SLIDER
1139
855
1311
888
plot-alpha
plot-alpha
0
1
0.1
.01
1
NIL
HORIZONTAL

SLIDER
8
760
180
793
ask-to-light-chance
ask-to-light-chance
0
100
24.0
1
1
%
HORIZONTAL

TEXTBOX
20
197
170
215
Agent parameters
11
0.0
1

SLIDER
13
504
195
537
reforest-chance
reforest-chance
0
100
31.0
1
1
%
HORIZONTAL

PLOT
776
634
1095
840
Economy value
NIL
NIL
0.0
10.0
0.0
10.0
true
true
"" ""
PENS
"Government" 1.0 0 -612749 true "" "plot government-money"
"Community" 1.0 0 -7500403 true "" "plot community-money"
"Base-line" 1.0 0 -2674135 true "" "plot 0"

SLIDER
590
632
762
665
Crops-selling-value
Crops-selling-value
0
100
43.0
1
1
NIL
HORIZONTAL

SLIDER
591
671
763
704
Logs-selling-value
Logs-selling-value
0
100
43.0
1
1
NIL
HORIZONTAL

SLIDER
590
710
762
743
taxes-logs
taxes-logs
0
0.25
0.25
0.01
1
NIL
HORIZONTAL

SLIDER
590
749
762
782
living-cost
living-cost
0
10
0.9
0.1
1
NIL
HORIZONTAL

SLIDER
590
787
762
820
ticks-th
ticks-th
0
100
6.0
1
1
NIL
HORIZONTAL

MONITOR
776
845
859
890
NIL
dead-agents
17
1
11

SLIDER
589
822
761
855
breeding-money-th
breeding-money-th
0
2500
2134.0
1
1
NIL
HORIZONTAL

MONITOR
868
846
970
891
NIL
breeded-agents
17
1
11

SLIDER
369
717
541
750
seed-cost
seed-cost
1
100
9.0
1
1
NIL
HORIZONTAL

SLIDER
587
903
689
936
breeding-cost
breeding-cost
0
100
75.0
1
1
NIL
HORIZONTAL

SLIDER
368
758
540
791
ignite-cost
ignite-cost
1
1000
103.0
1
1
NIL
HORIZONTAL

SLIDER
367
797
539
830
reforest-cost
reforest-cost
0
100
53.0
1
1
NIL
HORIZONTAL

SLIDER
590
862
694
895
taxes-crops
taxes-crops
0
0.25
0.25
0.01
1
NIL
HORIZONTAL

SLIDER
9
846
181
879
relocate-chance
relocate-chance
0
100
6.0
1
1
%
HORIZONTAL

SLIDER
195
848
367
881
max-agents
max-agents
100
500
271.0
1
1
NIL
HORIZONTAL

SLIDER
10
887
215
920
create-new-town-chance
create-new-town-chance
0
100
1.0
1
1
%
HORIZONTAL

SLIDER
583
946
755
979
gr-thresh
gr-thresh
0
.10
0.01
.001
1
NIL
HORIZONTAL

SLIDER
378
900
550
933
growth-smoothing
growth-smoothing
0
1
0.1
.01
1
NIL
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.2.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
