У┼
Јт
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ї
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
ї
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.6.22v2.6.1-9-gc2363d6d0258їц
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:*
dtype0
~
net_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namenet_output/kernel
w
%net_output/kernel/Read/ReadVariableOpReadVariableOpnet_output/kernel*
_output_shapes

:*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:*
dtype0
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
ѕ
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_10/kernel/m
Ђ
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

:*
dtype0
ѕ
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/m
Ђ
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:*
dtype0
ѕ
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_12/kernel/m
Ђ
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:*
dtype0
ѕ
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_13/kernel/m
Ђ
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:*
dtype0
ї
Adam/net_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/net_output/kernel/m
Ё
,Adam/net_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/net_output/kernel/m*
_output_shapes

:*
dtype0
ѕ
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_14/kernel/m
Ђ
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:*
dtype0
ѕ
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_15/kernel/m
Ђ
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes

:*
dtype0
ѕ
Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_16/kernel/m
Ђ
*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m*
_output_shapes

:*
dtype0
ѕ
Adam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_17/kernel/m
Ђ
*Adam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/m*
_output_shapes

:*
dtype0
ѕ
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_18/kernel/m
Ђ
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
_output_shapes

:*
dtype0
ѕ
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_19/kernel/m
Ђ
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes

:*
dtype0
ѕ
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_10/kernel/v
Ђ
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

:*
dtype0
ѕ
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/v
Ђ
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:*
dtype0
ѕ
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_12/kernel/v
Ђ
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:*
dtype0
ѕ
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_13/kernel/v
Ђ
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:*
dtype0
ї
Adam/net_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/net_output/kernel/v
Ё
,Adam/net_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/net_output/kernel/v*
_output_shapes

:*
dtype0
ѕ
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_14/kernel/v
Ђ
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:*
dtype0
ѕ
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_15/kernel/v
Ђ
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes

:*
dtype0
ѕ
Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_16/kernel/v
Ђ
*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v*
_output_shapes

:*
dtype0
ѕ
Adam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_17/kernel/v
Ђ
*Adam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/v*
_output_shapes

:*
dtype0
ѕ
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_18/kernel/v
Ђ
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes

:*
dtype0
ѕ
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_19/kernel/v
Ђ
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes

:*
dtype0

NoOpNoOp
В`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Д`
valueЮ`Bџ` BЊ`
┤
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer_with_weights-9
layer-20
layer-21
layer_with_weights-10
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
	optimizer
 loss
!	variables
"regularization_losses
#trainable_variables
$	keras_api
%
signatures
 
^

&kernel
'regularization_losses
(	variables
)trainable_variables
*	keras_api
^

+kernel
,regularization_losses
-	variables
.trainable_variables
/	keras_api
R
0regularization_losses
1	variables
2trainable_variables
3	keras_api
^

4kernel
5regularization_losses
6	variables
7trainable_variables
8	keras_api
R
9regularization_losses
:	variables
;trainable_variables
<	keras_api
^

=kernel
>regularization_losses
?	variables
@trainable_variables
A	keras_api
R
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
^

Fkernel
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api

K	keras_api

L	keras_api

M	keras_api
^

Nkernel
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
R
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
^

Wkernel
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
R
\regularization_losses
]	variables
^trainable_variables
_	keras_api
^

`kernel
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
R
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
^

ikernel
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
R
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
^

rkernel
sregularization_losses
t	variables
utrainable_variables
v	keras_api
R
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
^

{kernel
|regularization_losses
}	variables
~trainable_variables
	keras_api

ђ	keras_api

Ђ	keras_api

ѓ	keras_api

Ѓ	keras_api

ё	keras_api

Ё	keras_api
V
єregularization_losses
Є	variables
ѕtrainable_variables
Ѕ	keras_api
А
	іiter
Іbeta_1
їbeta_2

Їdecay
јlearning_rate&m§+m■4m =mђFmЂNmѓWmЃ`mёimЁrmє{mЄ&vѕ+vЅ4vі=vІFvїNvЇWvј`vЈivљrvЉ{vњ
 
N
&0
+1
42
=3
F4
N5
W6
`7
i8
r9
{10
 
N
&0
+1
42
=3
F4
N5
W6
`7
i8
r9
{10
▓
Јlayer_metrics
љlayers
!	variables
"regularization_losses
 Љlayer_regularization_losses
#trainable_variables
њnon_trainable_variables
Њmetrics
 
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

&0

&0
▓
ћlayer_metrics
Ћlayers
 ќlayer_regularization_losses
'regularization_losses
(	variables
)trainable_variables
Ќnon_trainable_variables
ўmetrics
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

+0

+0
▓
Ўlayer_metrics
џlayers
 Џlayer_regularization_losses
,regularization_losses
-	variables
.trainable_variables
юnon_trainable_variables
Юmetrics
 
 
 
▓
ъlayer_metrics
Ъlayers
 аlayer_regularization_losses
0regularization_losses
1	variables
2trainable_variables
Аnon_trainable_variables
бmetrics
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

40

40
▓
Бlayer_metrics
цlayers
 Цlayer_regularization_losses
5regularization_losses
6	variables
7trainable_variables
дnon_trainable_variables
Дmetrics
 
 
 
▓
еlayer_metrics
Еlayers
 фlayer_regularization_losses
9regularization_losses
:	variables
;trainable_variables
Фnon_trainable_variables
гmetrics
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

=0

=0
▓
Гlayer_metrics
«layers
 »layer_regularization_losses
>regularization_losses
?	variables
@trainable_variables
░non_trainable_variables
▒metrics
 
 
 
▓
▓layer_metrics
│layers
 ┤layer_regularization_losses
Bregularization_losses
C	variables
Dtrainable_variables
хnon_trainable_variables
Хmetrics
][
VARIABLE_VALUEnet_output/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

F0

F0
▓
иlayer_metrics
Иlayers
 ╣layer_regularization_losses
Gregularization_losses
H	variables
Itrainable_variables
║non_trainable_variables
╗metrics
 
 
 
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

N0

N0
▓
╝layer_metrics
йlayers
 Йlayer_regularization_losses
Oregularization_losses
P	variables
Qtrainable_variables
┐non_trainable_variables
└metrics
 
 
 
▓
┴layer_metrics
┬layers
 ├layer_regularization_losses
Sregularization_losses
T	variables
Utrainable_variables
─non_trainable_variables
┼metrics
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

W0

W0
▓
кlayer_metrics
Кlayers
 ╚layer_regularization_losses
Xregularization_losses
Y	variables
Ztrainable_variables
╔non_trainable_variables
╩metrics
 
 
 
▓
╦layer_metrics
╠layers
 ═layer_regularization_losses
\regularization_losses
]	variables
^trainable_variables
╬non_trainable_variables
¤metrics
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

`0

`0
▓
лlayer_metrics
Лlayers
 мlayer_regularization_losses
aregularization_losses
b	variables
ctrainable_variables
Мnon_trainable_variables
нmetrics
 
 
 
▓
Нlayer_metrics
оlayers
 Оlayer_regularization_losses
eregularization_losses
f	variables
gtrainable_variables
пnon_trainable_variables
┘metrics
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

i0

i0
▓
┌layer_metrics
█layers
 ▄layer_regularization_losses
jregularization_losses
k	variables
ltrainable_variables
Пnon_trainable_variables
яmetrics
 
 
 
▓
▀layer_metrics
Яlayers
 рlayer_regularization_losses
nregularization_losses
o	variables
ptrainable_variables
Рnon_trainable_variables
сmetrics
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

r0

r0
▓
Сlayer_metrics
тlayers
 Тlayer_regularization_losses
sregularization_losses
t	variables
utrainable_variables
уnon_trainable_variables
Уmetrics
 
 
 
▓
жlayer_metrics
Жlayers
 вlayer_regularization_losses
wregularization_losses
x	variables
ytrainable_variables
Вnon_trainable_variables
ьmetrics
\Z
VARIABLE_VALUEdense_19/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

{0

{0
▓
Ьlayer_metrics
№layers
 ­layer_regularization_losses
|regularization_losses
}	variables
~trainable_variables
ыnon_trainable_variables
Ыmetrics
 
 
 
 
 
 
 
 
 
х
зlayer_metrics
Зlayers
 шlayer_regularization_losses
єregularization_losses
Є	variables
ѕtrainable_variables
Шnon_trainable_variables
эmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
Т
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
 
 

Э0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

щtotal

Щcount
ч	variables
Ч	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

щ0
Щ1

ч	variables
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/net_output/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_17/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_19/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/net_output/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_17/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_19/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_2Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
 
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_10/kerneldense_11/kerneldense_12/kerneldense_13/kernelnet_output/kerneldense_14/kerneldense_15/kerneldense_16/kerneldense_17/kerneldense_18/kerneldense_19/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference_signature_wrapper_5867
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ё
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_10/kernel/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp%net_output/kernel/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp,Adam/net_output/kernel/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp*Adam/dense_17/kernel/m/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp,Adam/net_output/kernel/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp*Adam/dense_17/kernel/v/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOpConst*5
Tin.
,2*	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *&
f!R
__inference__traced_save_7065
Я
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10/kerneldense_11/kerneldense_12/kerneldense_13/kernelnet_output/kerneldense_14/kerneldense_15/kerneldense_16/kerneldense_17/kerneldense_18/kerneldense_19/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_10/kernel/mAdam/dense_11/kernel/mAdam/dense_12/kernel/mAdam/dense_13/kernel/mAdam/net_output/kernel/mAdam/dense_14/kernel/mAdam/dense_15/kernel/mAdam/dense_16/kernel/mAdam/dense_17/kernel/mAdam/dense_18/kernel/mAdam/dense_19/kernel/mAdam/dense_10/kernel/vAdam/dense_11/kernel/vAdam/dense_12/kernel/vAdam/dense_13/kernel/vAdam/net_output/kernel/vAdam/dense_14/kernel/vAdam/dense_15/kernel/vAdam/dense_16/kernel/vAdam/dense_17/kernel/vAdam/dense_18/kernel/vAdam/dense_19/kernel/v*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__traced_restore_7195Шо
с
E
.__inference_dense_14_activity_regularizer_4419
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
ы
b
D__inference_dropout_13_layer_call_and_return_conditional_losses_4699

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─
{
'__inference_dense_10_layer_call_fn_6447

inputs
unknown:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_44992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_15_layer_call_and_return_conditional_losses_6890

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
ф
F__inference_dense_14_layer_call_and_return_all_conditional_losses_6601

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_46282
StatefulPartitionedCallх
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_14_activity_regularizer_44192
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╝
E
)__inference_dropout_13_layer_call_fn_6716

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_46992
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_11_layer_call_and_return_conditional_losses_6850

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_12_layer_call_and_return_conditional_losses_4546

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ъ
b
)__inference_dropout_12_layer_call_fn_6678

inputs
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_12_layer_call_and_return_conditional_losses_50082
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
с
E
.__inference_dense_15_activity_regularizer_4432
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
с
E
.__inference_dense_19_activity_regularizer_4484
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
п
Ф
B__inference_dense_17_layer_call_and_return_conditional_losses_4709

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ъ
b
)__inference_dropout_13_layer_call_fn_6721

inputs
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_49672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
эб
║

A__inference_model_2_layer_call_and_return_conditional_losses_5832
input_2
dense_10_5673:
dense_11_5684:
dense_12_5696:
dense_13_5708:!
net_output_5720:
dense_14_5739:
dense_15_5751:
dense_16_5763:
dense_17_5775:
dense_18_5787:
dense_19_5799:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12ѕб dense_10/StatefulPartitionedCallб dense_11/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб dense_13/StatefulPartitionedCallб dense_14/StatefulPartitionedCallб dense_15/StatefulPartitionedCallб dense_16/StatefulPartitionedCallб dense_17/StatefulPartitionedCallб dense_18/StatefulPartitionedCallб dense_19/StatefulPartitionedCallб"dropout_10/StatefulPartitionedCallб"dropout_11/StatefulPartitionedCallб"dropout_12/StatefulPartitionedCallб"dropout_13/StatefulPartitionedCallб"dropout_14/StatefulPartitionedCallб"dropout_15/StatefulPartitionedCallб!dropout_8/StatefulPartitionedCallб!dropout_9/StatefulPartitionedCallб"net_output/StatefulPartitionedCall■
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_10_5673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_44992"
 dense_10/StatefulPartitionedCallЭ
,dense_10/ActivityRegularizer/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_10_activity_regularizer_43542.
,dense_10/ActivityRegularizer/PartitionedCallА
"dense_10/ActivityRegularizer/ShapeShape)dense_10/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_10/ActivityRegularizer/Shape«
0dense_10/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_10/ActivityRegularizer/strided_slice/stack▓
2dense_10/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_1▓
2dense_10/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_2љ
*dense_10/ActivityRegularizer/strided_sliceStridedSlice+dense_10/ActivityRegularizer/Shape:output:09dense_10/ActivityRegularizer/strided_slice/stack:output:0;dense_10/ActivityRegularizer/strided_slice/stack_1:output:0;dense_10/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_10/ActivityRegularizer/strided_slice│
!dense_10/ActivityRegularizer/CastCast3dense_10/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_10/ActivityRegularizer/Castо
$dense_10/ActivityRegularizer/truedivRealDiv5dense_10/ActivityRegularizer/PartitionedCall:output:0%dense_10/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_10/ActivityRegularizer/truedivа
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_5684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_45192"
 dense_11/StatefulPartitionedCallЭ
,dense_11/ActivityRegularizer/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_11_activity_regularizer_43672.
,dense_11/ActivityRegularizer/PartitionedCallА
"dense_11/ActivityRegularizer/ShapeShape)dense_11/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_11/ActivityRegularizer/Shape«
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_11/ActivityRegularizer/strided_slice/stack▓
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_1▓
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_2љ
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_11/ActivityRegularizer/strided_slice│
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_11/ActivityRegularizer/Castо
$dense_11/ActivityRegularizer/truedivRealDiv5dense_11/ActivityRegularizer/PartitionedCall:output:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_11/ActivityRegularizer/truedivљ
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_8_layer_call_and_return_conditional_losses_51902#
!dropout_8/StatefulPartitionedCallА
 dense_12/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_12_5696*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_45462"
 dense_12/StatefulPartitionedCallЭ
,dense_12/ActivityRegularizer/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_12_activity_regularizer_43802.
,dense_12/ActivityRegularizer/PartitionedCallА
"dense_12/ActivityRegularizer/ShapeShape)dense_12/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_12/ActivityRegularizer/Shape«
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_12/ActivityRegularizer/strided_slice/stack▓
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_1▓
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_2љ
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_12/ActivityRegularizer/strided_slice│
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_12/ActivityRegularizer/Castо
$dense_12/ActivityRegularizer/truedivRealDiv5dense_12/ActivityRegularizer/PartitionedCall:output:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_12/ActivityRegularizer/truediv┤
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_9_layer_call_and_return_conditional_losses_51492#
!dropout_9/StatefulPartitionedCallА
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_13_5708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_45732"
 dense_13/StatefulPartitionedCallЭ
,dense_13/ActivityRegularizer/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_13_activity_regularizer_43932.
,dense_13/ActivityRegularizer/PartitionedCallА
"dense_13/ActivityRegularizer/ShapeShape)dense_13/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_13/ActivityRegularizer/Shape«
0dense_13/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_13/ActivityRegularizer/strided_slice/stack▓
2dense_13/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_1▓
2dense_13/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_2љ
*dense_13/ActivityRegularizer/strided_sliceStridedSlice+dense_13/ActivityRegularizer/Shape:output:09dense_13/ActivityRegularizer/strided_slice/stack:output:0;dense_13/ActivityRegularizer/strided_slice/stack_1:output:0;dense_13/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_13/ActivityRegularizer/strided_slice│
!dense_13/ActivityRegularizer/CastCast3dense_13/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_13/ActivityRegularizer/Castо
$dense_13/ActivityRegularizer/truedivRealDiv5dense_13/ActivityRegularizer/PartitionedCall:output:0%dense_13/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_13/ActivityRegularizer/truedivи
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_51082$
"dropout_10/StatefulPartitionedCallф
"net_output/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0net_output_5720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_net_output_layer_call_and_return_conditional_losses_46002$
"net_output/StatefulPartitionedCallђ
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *9
f4R2
0__inference_net_output_activity_regularizer_440620
.net_output/ActivityRegularizer/PartitionedCallД
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2&
$net_output/ActivityRegularizer/Shape▓
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2net_output/ActivityRegularizer/strided_slice/stackХ
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_1Х
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_2ю
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,net_output/ActivityRegularizer/strided_slice╣
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#net_output/ActivityRegularizer/Castя
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&net_output/ActivityRegularizer/truedivЂ
tf.math.subtract_2/Sub/yConst*
_output_shapes
:*
dtype0*
valueB*    2
tf.math.subtract_2/Sub/y╣
tf.math.subtract_2/SubSub+net_output/StatefulPartitionedCall:output:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:         2
tf.math.subtract_2/Subo
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.pow_1/Pow/yЎ
tf.math.pow_1/PowPowtf.math.subtract_2/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*'
_output_shapes
:         2
tf.math.pow_1/PowБ
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indicesх
tf.math.reduce_sum_1/SumSumtf.math.pow_1/Pow:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_1/Sumё
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constф
tf.math.reduce_mean_2/MeanMean!tf.math.reduce_sum_1/Sum:output:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/Meanб
 dense_14/StatefulPartitionedCallStatefulPartitionedCall+net_output/StatefulPartitionedCall:output:0dense_14_5739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_46282"
 dense_14/StatefulPartitionedCallЭ
,dense_14/ActivityRegularizer/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_14_activity_regularizer_44192.
,dense_14/ActivityRegularizer/PartitionedCallА
"dense_14/ActivityRegularizer/ShapeShape)dense_14/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_14/ActivityRegularizer/Shape«
0dense_14/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_14/ActivityRegularizer/strided_slice/stack▓
2dense_14/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_1▓
2dense_14/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_2љ
*dense_14/ActivityRegularizer/strided_sliceStridedSlice+dense_14/ActivityRegularizer/Shape:output:09dense_14/ActivityRegularizer/strided_slice/stack:output:0;dense_14/ActivityRegularizer/strided_slice/stack_1:output:0;dense_14/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_14/ActivityRegularizer/strided_slice│
!dense_14/ActivityRegularizer/CastCast3dense_14/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_14/ActivityRegularizer/Castо
$dense_14/ActivityRegularizer/truedivRealDiv5dense_14/ActivityRegularizer/PartitionedCall:output:0%dense_14/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_14/ActivityRegularizer/truedivИ
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_50492$
"dropout_11/StatefulPartitionedCallб
 dense_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_15_5751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_46552"
 dense_15/StatefulPartitionedCallЭ
,dense_15/ActivityRegularizer/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_15_activity_regularizer_44322.
,dense_15/ActivityRegularizer/PartitionedCallА
"dense_15/ActivityRegularizer/ShapeShape)dense_15/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_15/ActivityRegularizer/Shape«
0dense_15/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_15/ActivityRegularizer/strided_slice/stack▓
2dense_15/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_1▓
2dense_15/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_2љ
*dense_15/ActivityRegularizer/strided_sliceStridedSlice+dense_15/ActivityRegularizer/Shape:output:09dense_15/ActivityRegularizer/strided_slice/stack:output:0;dense_15/ActivityRegularizer/strided_slice/stack_1:output:0;dense_15/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_15/ActivityRegularizer/strided_slice│
!dense_15/ActivityRegularizer/CastCast3dense_15/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_15/ActivityRegularizer/Castо
$dense_15/ActivityRegularizer/truedivRealDiv5dense_15/ActivityRegularizer/PartitionedCall:output:0%dense_15/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_15/ActivityRegularizer/truedivИ
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_12_layer_call_and_return_conditional_losses_50082$
"dropout_12/StatefulPartitionedCallб
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_16_5763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_46822"
 dense_16/StatefulPartitionedCallЭ
,dense_16/ActivityRegularizer/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_16_activity_regularizer_44452.
,dense_16/ActivityRegularizer/PartitionedCallА
"dense_16/ActivityRegularizer/ShapeShape)dense_16/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape«
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack▓
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1▓
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2љ
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice│
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Castо
$dense_16/ActivityRegularizer/truedivRealDiv5dense_16/ActivityRegularizer/PartitionedCall:output:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_16/ActivityRegularizer/truedivИ
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_49672$
"dropout_13/StatefulPartitionedCallб
 dense_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_17_5775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_47092"
 dense_17/StatefulPartitionedCallЭ
,dense_17/ActivityRegularizer/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_17_activity_regularizer_44582.
,dense_17/ActivityRegularizer/PartitionedCallА
"dense_17/ActivityRegularizer/ShapeShape)dense_17/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_17/ActivityRegularizer/Shape«
0dense_17/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_17/ActivityRegularizer/strided_slice/stack▓
2dense_17/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_1▓
2dense_17/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_2љ
*dense_17/ActivityRegularizer/strided_sliceStridedSlice+dense_17/ActivityRegularizer/Shape:output:09dense_17/ActivityRegularizer/strided_slice/stack:output:0;dense_17/ActivityRegularizer/strided_slice/stack_1:output:0;dense_17/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_17/ActivityRegularizer/strided_slice│
!dense_17/ActivityRegularizer/CastCast3dense_17/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_17/ActivityRegularizer/Castо
$dense_17/ActivityRegularizer/truedivRealDiv5dense_17/ActivityRegularizer/PartitionedCall:output:0%dense_17/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_17/ActivityRegularizer/truedivИ
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0#^dropout_13/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_14_layer_call_and_return_conditional_losses_49262$
"dropout_14/StatefulPartitionedCallб
 dense_18/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0dense_18_5787*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_47362"
 dense_18/StatefulPartitionedCallЭ
,dense_18/ActivityRegularizer/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_18_activity_regularizer_44712.
,dense_18/ActivityRegularizer/PartitionedCallА
"dense_18/ActivityRegularizer/ShapeShape)dense_18/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_18/ActivityRegularizer/Shape«
0dense_18/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_18/ActivityRegularizer/strided_slice/stack▓
2dense_18/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_1▓
2dense_18/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_2љ
*dense_18/ActivityRegularizer/strided_sliceStridedSlice+dense_18/ActivityRegularizer/Shape:output:09dense_18/ActivityRegularizer/strided_slice/stack:output:0;dense_18/ActivityRegularizer/strided_slice/stack_1:output:0;dense_18/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_18/ActivityRegularizer/strided_slice│
!dense_18/ActivityRegularizer/CastCast3dense_18/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_18/ActivityRegularizer/Castо
$dense_18/ActivityRegularizer/truedivRealDiv5dense_18/ActivityRegularizer/PartitionedCall:output:0%dense_18/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_18/ActivityRegularizer/truedivИ
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_15_layer_call_and_return_conditional_losses_48852$
"dropout_15/StatefulPartitionedCallб
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_19_5799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_47632"
 dense_19/StatefulPartitionedCallЭ
,dense_19/ActivityRegularizer/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_19_activity_regularizer_44842.
,dense_19/ActivityRegularizer/PartitionedCallА
"dense_19/ActivityRegularizer/ShapeShape)dense_19/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_19/ActivityRegularizer/Shape«
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_19/ActivityRegularizer/strided_slice/stack▓
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_1▓
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_2љ
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_19/ActivityRegularizer/strided_slice│
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_19/ActivityRegularizer/Castо
$dense_19/ActivityRegularizer/truedivRealDiv5dense_19/ActivityRegularizer/PartitionedCall:output:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_19/ActivityRegularizer/truedivЮ
tf.math.subtract_3/SubSub)dense_19/StatefulPartitionedCall:output:0input_2*
T0*'
_output_shapes
:         2
tf.math.subtract_3/Subі
tf.math.square_1/SquareSquaretf.math.subtract_3/Sub:z:0*
T0*'
_output_shapes
:         2
tf.math.square_1/SquareІ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constц
tf.math.reduce_mean_3/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean░
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_2/Mean:output:0#tf.math.reduce_mean_3/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_2/AddV2y
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 */Lь62
tf.__operators__.add_3/yФ
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_3/AddV2С
add_loss_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_add_loss_1_layer_call_and_return_conditional_losses_47872
add_loss_1/PartitionedCallx
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityv

Identity_1Identity(dense_10/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1v

Identity_2Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_2v

Identity_3Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3v

Identity_4Identity(dense_13/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_4x

Identity_5Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_5v

Identity_6Identity(dense_14/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_6v

Identity_7Identity(dense_15/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_7v

Identity_8Identity(dense_16/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_8v

Identity_9Identity(dense_17/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_9x
Identity_10Identity(dense_18/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_10x
Identity_11Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_11s
Identity_12Identity#add_loss_1/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 2
Identity_12э
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
┌
Г
D__inference_net_output_layer_call_and_return_conditional_losses_4600

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╝
E
)__inference_dropout_10_layer_call_fn_6571

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_45902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
│»
Џ

A__inference_model_2_layer_call_and_return_conditional_losses_6082

inputs9
'dense_10_matmul_readvariableop_resource:9
'dense_11_matmul_readvariableop_resource:9
'dense_12_matmul_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:;
)net_output_matmul_readvariableop_resource:9
'dense_14_matmul_readvariableop_resource:9
'dense_15_matmul_readvariableop_resource:9
'dense_16_matmul_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:9
'dense_18_matmul_readvariableop_resource:9
'dense_19_matmul_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12ѕбdense_10/MatMul/ReadVariableOpбdense_11/MatMul/ReadVariableOpбdense_12/MatMul/ReadVariableOpбdense_13/MatMul/ReadVariableOpбdense_14/MatMul/ReadVariableOpбdense_15/MatMul/ReadVariableOpбdense_16/MatMul/ReadVariableOpбdense_17/MatMul/ReadVariableOpбdense_18/MatMul/ReadVariableOpбdense_19/MatMul/ReadVariableOpб net_output/MatMul/ReadVariableOpе
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_10/MatMul/ReadVariableOpј
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_10/MatMuls
dense_10/ReluReludense_10/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_10/ReluБ
#dense_10/ActivityRegularizer/SquareSquaredense_10/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_10/ActivityRegularizer/SquareЎ
"dense_10/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_10/ActivityRegularizer/Const┬
 dense_10/ActivityRegularizer/SumSum'dense_10/ActivityRegularizer/Square:y:0+dense_10/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_10/ActivityRegularizer/SumЇ
"dense_10/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_10/ActivityRegularizer/mul/x─
 dense_10/ActivityRegularizer/mulMul+dense_10/ActivityRegularizer/mul/x:output:0)dense_10/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_10/ActivityRegularizer/mulЊ
"dense_10/ActivityRegularizer/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_10/ActivityRegularizer/Shape«
0dense_10/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_10/ActivityRegularizer/strided_slice/stack▓
2dense_10/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_1▓
2dense_10/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_2љ
*dense_10/ActivityRegularizer/strided_sliceStridedSlice+dense_10/ActivityRegularizer/Shape:output:09dense_10/ActivityRegularizer/strided_slice/stack:output:0;dense_10/ActivityRegularizer/strided_slice/stack_1:output:0;dense_10/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_10/ActivityRegularizer/strided_slice│
!dense_10/ActivityRegularizer/CastCast3dense_10/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_10/ActivityRegularizer/Cast┼
$dense_10/ActivityRegularizer/truedivRealDiv$dense_10/ActivityRegularizer/mul:z:0%dense_10/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_10/ActivityRegularizer/truedivе
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOpБ
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/MatMuls
dense_11/ReluReludense_11/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_11/ReluБ
#dense_11/ActivityRegularizer/SquareSquaredense_11/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_11/ActivityRegularizer/SquareЎ
"dense_11/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_11/ActivityRegularizer/Const┬
 dense_11/ActivityRegularizer/SumSum'dense_11/ActivityRegularizer/Square:y:0+dense_11/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_11/ActivityRegularizer/SumЇ
"dense_11/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_11/ActivityRegularizer/mul/x─
 dense_11/ActivityRegularizer/mulMul+dense_11/ActivityRegularizer/mul/x:output:0)dense_11/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_11/ActivityRegularizer/mulЊ
"dense_11/ActivityRegularizer/ShapeShapedense_11/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_11/ActivityRegularizer/Shape«
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_11/ActivityRegularizer/strided_slice/stack▓
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_1▓
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_2љ
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_11/ActivityRegularizer/strided_slice│
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_11/ActivityRegularizer/Cast┼
$dense_11/ActivityRegularizer/truedivRealDiv$dense_11/ActivityRegularizer/mul:z:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_11/ActivityRegularizer/truedivЃ
dropout_8/IdentityIdentitydense_11/Relu:activations:0*
T0*'
_output_shapes
:         2
dropout_8/Identityе
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOpБ
dense_12/MatMulMatMuldropout_8/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_12/MatMuls
dense_12/ReluReludense_12/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_12/ReluБ
#dense_12/ActivityRegularizer/SquareSquaredense_12/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_12/ActivityRegularizer/SquareЎ
"dense_12/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_12/ActivityRegularizer/Const┬
 dense_12/ActivityRegularizer/SumSum'dense_12/ActivityRegularizer/Square:y:0+dense_12/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_12/ActivityRegularizer/SumЇ
"dense_12/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_12/ActivityRegularizer/mul/x─
 dense_12/ActivityRegularizer/mulMul+dense_12/ActivityRegularizer/mul/x:output:0)dense_12/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_12/ActivityRegularizer/mulЊ
"dense_12/ActivityRegularizer/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_12/ActivityRegularizer/Shape«
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_12/ActivityRegularizer/strided_slice/stack▓
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_1▓
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_2љ
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_12/ActivityRegularizer/strided_slice│
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_12/ActivityRegularizer/Cast┼
$dense_12/ActivityRegularizer/truedivRealDiv$dense_12/ActivityRegularizer/mul:z:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_12/ActivityRegularizer/truedivЃ
dropout_9/IdentityIdentitydense_12/Relu:activations:0*
T0*'
_output_shapes
:         2
dropout_9/Identityе
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_13/MatMul/ReadVariableOpБ
dense_13/MatMulMatMuldropout_9/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/MatMuls
dense_13/ReluReludense_13/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_13/ReluБ
#dense_13/ActivityRegularizer/SquareSquaredense_13/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_13/ActivityRegularizer/SquareЎ
"dense_13/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_13/ActivityRegularizer/Const┬
 dense_13/ActivityRegularizer/SumSum'dense_13/ActivityRegularizer/Square:y:0+dense_13/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_13/ActivityRegularizer/SumЇ
"dense_13/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_13/ActivityRegularizer/mul/x─
 dense_13/ActivityRegularizer/mulMul+dense_13/ActivityRegularizer/mul/x:output:0)dense_13/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_13/ActivityRegularizer/mulЊ
"dense_13/ActivityRegularizer/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_13/ActivityRegularizer/Shape«
0dense_13/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_13/ActivityRegularizer/strided_slice/stack▓
2dense_13/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_1▓
2dense_13/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_2љ
*dense_13/ActivityRegularizer/strided_sliceStridedSlice+dense_13/ActivityRegularizer/Shape:output:09dense_13/ActivityRegularizer/strided_slice/stack:output:0;dense_13/ActivityRegularizer/strided_slice/stack_1:output:0;dense_13/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_13/ActivityRegularizer/strided_slice│
!dense_13/ActivityRegularizer/CastCast3dense_13/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_13/ActivityRegularizer/Cast┼
$dense_13/ActivityRegularizer/truedivRealDiv$dense_13/ActivityRegularizer/mul:z:0%dense_13/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_13/ActivityRegularizer/truedivЁ
dropout_10/IdentityIdentitydense_13/Relu:activations:0*
T0*'
_output_shapes
:         2
dropout_10/Identity«
 net_output/MatMul/ReadVariableOpReadVariableOp)net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 net_output/MatMul/ReadVariableOpф
net_output/MatMulMatMuldropout_10/Identity:output:0(net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
net_output/MatMuly
net_output/ReluRelunet_output/MatMul:product:0*
T0*'
_output_shapes
:         2
net_output/ReluЕ
%net_output/ActivityRegularizer/SquareSquarenet_output/Relu:activations:0*
T0*'
_output_shapes
:         2'
%net_output/ActivityRegularizer/SquareЮ
$net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$net_output/ActivityRegularizer/Const╩
"net_output/ActivityRegularizer/SumSum)net_output/ActivityRegularizer/Square:y:0-net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2$
"net_output/ActivityRegularizer/SumЉ
$net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2&
$net_output/ActivityRegularizer/mul/x╠
"net_output/ActivityRegularizer/mulMul-net_output/ActivityRegularizer/mul/x:output:0+net_output/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"net_output/ActivityRegularizer/mulЎ
$net_output/ActivityRegularizer/ShapeShapenet_output/Relu:activations:0*
T0*
_output_shapes
:2&
$net_output/ActivityRegularizer/Shape▓
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2net_output/ActivityRegularizer/strided_slice/stackХ
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_1Х
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_2ю
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,net_output/ActivityRegularizer/strided_slice╣
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#net_output/ActivityRegularizer/Cast═
&net_output/ActivityRegularizer/truedivRealDiv&net_output/ActivityRegularizer/mul:z:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&net_output/ActivityRegularizer/truedivЂ
tf.math.subtract_2/Sub/yConst*
_output_shapes
:*
dtype0*
valueB*    2
tf.math.subtract_2/Sub/yФ
tf.math.subtract_2/SubSubnet_output/Relu:activations:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:         2
tf.math.subtract_2/Subo
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.pow_1/Pow/yЎ
tf.math.pow_1/PowPowtf.math.subtract_2/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*'
_output_shapes
:         2
tf.math.pow_1/PowБ
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indicesх
tf.math.reduce_sum_1/SumSumtf.math.pow_1/Pow:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_1/Sumё
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constф
tf.math.reduce_mean_2/MeanMean!tf.math.reduce_sum_1/Sum:output:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/Meanе
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_14/MatMul/ReadVariableOpЦ
dense_14/MatMulMatMulnet_output/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMuls
dense_14/ReluReludense_14/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_14/ReluБ
#dense_14/ActivityRegularizer/SquareSquaredense_14/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_14/ActivityRegularizer/SquareЎ
"dense_14/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_14/ActivityRegularizer/Const┬
 dense_14/ActivityRegularizer/SumSum'dense_14/ActivityRegularizer/Square:y:0+dense_14/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_14/ActivityRegularizer/SumЇ
"dense_14/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_14/ActivityRegularizer/mul/x─
 dense_14/ActivityRegularizer/mulMul+dense_14/ActivityRegularizer/mul/x:output:0)dense_14/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_14/ActivityRegularizer/mulЊ
"dense_14/ActivityRegularizer/ShapeShapedense_14/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_14/ActivityRegularizer/Shape«
0dense_14/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_14/ActivityRegularizer/strided_slice/stack▓
2dense_14/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_1▓
2dense_14/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_2љ
*dense_14/ActivityRegularizer/strided_sliceStridedSlice+dense_14/ActivityRegularizer/Shape:output:09dense_14/ActivityRegularizer/strided_slice/stack:output:0;dense_14/ActivityRegularizer/strided_slice/stack_1:output:0;dense_14/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_14/ActivityRegularizer/strided_slice│
!dense_14/ActivityRegularizer/CastCast3dense_14/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_14/ActivityRegularizer/Cast┼
$dense_14/ActivityRegularizer/truedivRealDiv$dense_14/ActivityRegularizer/mul:z:0%dense_14/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_14/ActivityRegularizer/truedivЁ
dropout_11/IdentityIdentitydense_14/Relu:activations:0*
T0*'
_output_shapes
:         2
dropout_11/Identityе
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_15/MatMul/ReadVariableOpц
dense_15/MatMulMatMuldropout_11/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/MatMuls
dense_15/ReluReludense_15/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_15/ReluБ
#dense_15/ActivityRegularizer/SquareSquaredense_15/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_15/ActivityRegularizer/SquareЎ
"dense_15/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_15/ActivityRegularizer/Const┬
 dense_15/ActivityRegularizer/SumSum'dense_15/ActivityRegularizer/Square:y:0+dense_15/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_15/ActivityRegularizer/SumЇ
"dense_15/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_15/ActivityRegularizer/mul/x─
 dense_15/ActivityRegularizer/mulMul+dense_15/ActivityRegularizer/mul/x:output:0)dense_15/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_15/ActivityRegularizer/mulЊ
"dense_15/ActivityRegularizer/ShapeShapedense_15/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_15/ActivityRegularizer/Shape«
0dense_15/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_15/ActivityRegularizer/strided_slice/stack▓
2dense_15/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_1▓
2dense_15/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_2љ
*dense_15/ActivityRegularizer/strided_sliceStridedSlice+dense_15/ActivityRegularizer/Shape:output:09dense_15/ActivityRegularizer/strided_slice/stack:output:0;dense_15/ActivityRegularizer/strided_slice/stack_1:output:0;dense_15/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_15/ActivityRegularizer/strided_slice│
!dense_15/ActivityRegularizer/CastCast3dense_15/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_15/ActivityRegularizer/Cast┼
$dense_15/ActivityRegularizer/truedivRealDiv$dense_15/ActivityRegularizer/mul:z:0%dense_15/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_15/ActivityRegularizer/truedivЁ
dropout_12/IdentityIdentitydense_15/Relu:activations:0*
T0*'
_output_shapes
:         2
dropout_12/Identityе
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_16/MatMul/ReadVariableOpц
dense_16/MatMulMatMuldropout_12/Identity:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_16/MatMuls
dense_16/ReluReludense_16/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_16/ReluБ
#dense_16/ActivityRegularizer/SquareSquaredense_16/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_16/ActivityRegularizer/SquareЎ
"dense_16/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_16/ActivityRegularizer/Const┬
 dense_16/ActivityRegularizer/SumSum'dense_16/ActivityRegularizer/Square:y:0+dense_16/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_16/ActivityRegularizer/SumЇ
"dense_16/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_16/ActivityRegularizer/mul/x─
 dense_16/ActivityRegularizer/mulMul+dense_16/ActivityRegularizer/mul/x:output:0)dense_16/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_16/ActivityRegularizer/mulЊ
"dense_16/ActivityRegularizer/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape«
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack▓
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1▓
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2љ
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice│
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Cast┼
$dense_16/ActivityRegularizer/truedivRealDiv$dense_16/ActivityRegularizer/mul:z:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_16/ActivityRegularizer/truedivЁ
dropout_13/IdentityIdentitydense_16/Relu:activations:0*
T0*'
_output_shapes
:         2
dropout_13/Identityе
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOpц
dense_17/MatMulMatMuldropout_13/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/MatMuls
dense_17/ReluReludense_17/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_17/ReluБ
#dense_17/ActivityRegularizer/SquareSquaredense_17/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_17/ActivityRegularizer/SquareЎ
"dense_17/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_17/ActivityRegularizer/Const┬
 dense_17/ActivityRegularizer/SumSum'dense_17/ActivityRegularizer/Square:y:0+dense_17/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_17/ActivityRegularizer/SumЇ
"dense_17/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_17/ActivityRegularizer/mul/x─
 dense_17/ActivityRegularizer/mulMul+dense_17/ActivityRegularizer/mul/x:output:0)dense_17/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_17/ActivityRegularizer/mulЊ
"dense_17/ActivityRegularizer/ShapeShapedense_17/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_17/ActivityRegularizer/Shape«
0dense_17/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_17/ActivityRegularizer/strided_slice/stack▓
2dense_17/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_1▓
2dense_17/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_2љ
*dense_17/ActivityRegularizer/strided_sliceStridedSlice+dense_17/ActivityRegularizer/Shape:output:09dense_17/ActivityRegularizer/strided_slice/stack:output:0;dense_17/ActivityRegularizer/strided_slice/stack_1:output:0;dense_17/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_17/ActivityRegularizer/strided_slice│
!dense_17/ActivityRegularizer/CastCast3dense_17/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_17/ActivityRegularizer/Cast┼
$dense_17/ActivityRegularizer/truedivRealDiv$dense_17/ActivityRegularizer/mul:z:0%dense_17/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_17/ActivityRegularizer/truedivЁ
dropout_14/IdentityIdentitydense_17/Relu:activations:0*
T0*'
_output_shapes
:         2
dropout_14/Identityе
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_18/MatMul/ReadVariableOpц
dense_18/MatMulMatMuldropout_14/Identity:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_18/MatMuls
dense_18/ReluReludense_18/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_18/ReluБ
#dense_18/ActivityRegularizer/SquareSquaredense_18/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_18/ActivityRegularizer/SquareЎ
"dense_18/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_18/ActivityRegularizer/Const┬
 dense_18/ActivityRegularizer/SumSum'dense_18/ActivityRegularizer/Square:y:0+dense_18/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_18/ActivityRegularizer/SumЇ
"dense_18/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_18/ActivityRegularizer/mul/x─
 dense_18/ActivityRegularizer/mulMul+dense_18/ActivityRegularizer/mul/x:output:0)dense_18/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_18/ActivityRegularizer/mulЊ
"dense_18/ActivityRegularizer/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_18/ActivityRegularizer/Shape«
0dense_18/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_18/ActivityRegularizer/strided_slice/stack▓
2dense_18/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_1▓
2dense_18/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_2љ
*dense_18/ActivityRegularizer/strided_sliceStridedSlice+dense_18/ActivityRegularizer/Shape:output:09dense_18/ActivityRegularizer/strided_slice/stack:output:0;dense_18/ActivityRegularizer/strided_slice/stack_1:output:0;dense_18/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_18/ActivityRegularizer/strided_slice│
!dense_18/ActivityRegularizer/CastCast3dense_18/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_18/ActivityRegularizer/Cast┼
$dense_18/ActivityRegularizer/truedivRealDiv$dense_18/ActivityRegularizer/mul:z:0%dense_18/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_18/ActivityRegularizer/truedivЁ
dropout_15/IdentityIdentitydense_18/Relu:activations:0*
T0*'
_output_shapes
:         2
dropout_15/Identityе
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_19/MatMul/ReadVariableOpц
dense_19/MatMulMatMuldropout_15/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_19/MatMul|
dense_19/SigmoidSigmoiddense_19/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_19/Sigmoidю
#dense_19/ActivityRegularizer/SquareSquaredense_19/Sigmoid:y:0*
T0*'
_output_shapes
:         2%
#dense_19/ActivityRegularizer/SquareЎ
"dense_19/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_19/ActivityRegularizer/Const┬
 dense_19/ActivityRegularizer/SumSum'dense_19/ActivityRegularizer/Square:y:0+dense_19/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_19/ActivityRegularizer/SumЇ
"dense_19/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_19/ActivityRegularizer/mul/x─
 dense_19/ActivityRegularizer/mulMul+dense_19/ActivityRegularizer/mul/x:output:0)dense_19/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_19/ActivityRegularizer/mulї
"dense_19/ActivityRegularizer/ShapeShapedense_19/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_19/ActivityRegularizer/Shape«
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_19/ActivityRegularizer/strided_slice/stack▓
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_1▓
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_2љ
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_19/ActivityRegularizer/strided_slice│
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_19/ActivityRegularizer/Cast┼
$dense_19/ActivityRegularizer/truedivRealDiv$dense_19/ActivityRegularizer/mul:z:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_19/ActivityRegularizer/truedivЄ
tf.math.subtract_3/SubSubdense_19/Sigmoid:y:0inputs*
T0*'
_output_shapes
:         2
tf.math.subtract_3/Subі
tf.math.square_1/SquareSquaretf.math.subtract_3/Sub:z:0*
T0*'
_output_shapes
:         2
tf.math.square_1/SquareІ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constц
tf.math.reduce_mean_3/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean░
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_2/Mean:output:0#tf.math.reduce_mean_3/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_2/AddV2y
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 */Lь62
tf.__operators__.add_3/yФ
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_3/AddV2x
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityv

Identity_1Identity(dense_10/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1v

Identity_2Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_2v

Identity_3Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3v

Identity_4Identity(dense_13/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_4x

Identity_5Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_5v

Identity_6Identity(dense_14/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_6v

Identity_7Identity(dense_15/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_7v

Identity_8Identity(dense_16/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_8v

Identity_9Identity(dense_17/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_9x
Identity_10Identity(dense_18/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_10x
Identity_11Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_11p
Identity_12Identity tf.__operators__.add_3/AddV2:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_12╗
NoOpNoOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_12/MatMul/ReadVariableOp^dense_13/MatMul/ReadVariableOp^dense_14/MatMul/ReadVariableOp^dense_15/MatMul/ReadVariableOp^dense_16/MatMul/ReadVariableOp^dense_17/MatMul/ReadVariableOp^dense_18/MatMul/ReadVariableOp^dense_19/MatMul/ReadVariableOp!^net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2D
 net_output/MatMul/ReadVariableOp net_output/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_12_layer_call_and_return_conditional_losses_5008

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_11_layer_call_and_return_conditional_losses_4645

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╚
}
)__inference_net_output_layer_call_fn_6592

inputs
unknown:
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_net_output_layer_call_and_return_conditional_losses_46002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
с
E
.__inference_dense_16_activity_regularizer_4445
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
─
{
'__inference_dense_13_layer_call_fn_6549

inputs
unknown:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_45732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
с
E
.__inference_dense_12_activity_regularizer_4380
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
ЁW
Т
__inference__traced_save_7065
file_prefix.
*savev2_dense_10_kernel_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop0
,savev2_net_output_kernel_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop7
3savev2_adam_net_output_kernel_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop5
1savev2_adam_dense_17_kernel_m_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop7
3savev2_adam_net_output_kernel_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop5
1savev2_adam_dense_17_kernel_v_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЋ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*Д
valueЮBџ)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names┌
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЙ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_10_kernel_read_readvariableop*savev2_dense_11_kernel_read_readvariableop*savev2_dense_12_kernel_read_readvariableop*savev2_dense_13_kernel_read_readvariableop,savev2_net_output_kernel_read_readvariableop*savev2_dense_14_kernel_read_readvariableop*savev2_dense_15_kernel_read_readvariableop*savev2_dense_16_kernel_read_readvariableop*savev2_dense_17_kernel_read_readvariableop*savev2_dense_18_kernel_read_readvariableop*savev2_dense_19_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop3savev2_adam_net_output_kernel_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop1savev2_adam_dense_17_kernel_m_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop3savev2_adam_net_output_kernel_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop1savev2_adam_dense_17_kernel_v_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*ы
_input_shapes▀
▄: :::::::::::: : : : : : : ::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$	 

_output_shapes

::$
 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$  

_output_shapes

::$! 

_output_shapes

::$" 

_output_shapes

::$# 

_output_shapes

::$$ 

_output_shapes

::$% 

_output_shapes

::$& 

_output_shapes

::$' 

_output_shapes

::$( 

_output_shapes

::)

_output_shapes
: 
и
ф
F__inference_dense_19_layer_call_and_return_all_conditional_losses_6816

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_47632
StatefulPartitionedCallх
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_19_activity_regularizer_44842
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
ф
F__inference_dense_17_layer_call_and_return_all_conditional_losses_6730

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_47092
StatefulPartitionedCallх
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_17_activity_regularizer_44582
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
с
E
.__inference_dense_18_activity_regularizer_4471
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
и
ф
F__inference_dense_12_layer_call_and_return_all_conditional_losses_6499

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_45462
StatefulPartitionedCallх
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_12_activity_regularizer_43802
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_18_layer_call_and_return_conditional_losses_4736

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
¤
p
D__inference_add_loss_1_layer_call_and_return_conditional_losses_4787

inputs
identity

identity_1I
IdentityIdentityinputs*
T0*
_output_shapes
: 2

IdentityM

Identity_1Identityinputs*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
─
{
'__inference_dense_14_layer_call_fn_6608

inputs
unknown:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_46282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
т
G
0__inference_net_output_activity_regularizer_4406
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
п
Ф
B__inference_dense_17_layer_call_and_return_conditional_losses_6906

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┌Ћ
Њ
A__inference_model_2_layer_call_and_return_conditional_losses_4803

inputs
dense_10_4500:
dense_11_4520:
dense_12_4547:
dense_13_4574:!
net_output_4601:
dense_14_4629:
dense_15_4656:
dense_16_4683:
dense_17_4710:
dense_18_4737:
dense_19_4764:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12ѕб dense_10/StatefulPartitionedCallб dense_11/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб dense_13/StatefulPartitionedCallб dense_14/StatefulPartitionedCallб dense_15/StatefulPartitionedCallб dense_16/StatefulPartitionedCallб dense_17/StatefulPartitionedCallб dense_18/StatefulPartitionedCallб dense_19/StatefulPartitionedCallб"net_output/StatefulPartitionedCall§
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_4500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_44992"
 dense_10/StatefulPartitionedCallЭ
,dense_10/ActivityRegularizer/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_10_activity_regularizer_43542.
,dense_10/ActivityRegularizer/PartitionedCallА
"dense_10/ActivityRegularizer/ShapeShape)dense_10/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_10/ActivityRegularizer/Shape«
0dense_10/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_10/ActivityRegularizer/strided_slice/stack▓
2dense_10/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_1▓
2dense_10/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_2љ
*dense_10/ActivityRegularizer/strided_sliceStridedSlice+dense_10/ActivityRegularizer/Shape:output:09dense_10/ActivityRegularizer/strided_slice/stack:output:0;dense_10/ActivityRegularizer/strided_slice/stack_1:output:0;dense_10/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_10/ActivityRegularizer/strided_slice│
!dense_10/ActivityRegularizer/CastCast3dense_10/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_10/ActivityRegularizer/Castо
$dense_10/ActivityRegularizer/truedivRealDiv5dense_10/ActivityRegularizer/PartitionedCall:output:0%dense_10/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_10/ActivityRegularizer/truedivа
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_4520*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_45192"
 dense_11/StatefulPartitionedCallЭ
,dense_11/ActivityRegularizer/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_11_activity_regularizer_43672.
,dense_11/ActivityRegularizer/PartitionedCallА
"dense_11/ActivityRegularizer/ShapeShape)dense_11/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_11/ActivityRegularizer/Shape«
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_11/ActivityRegularizer/strided_slice/stack▓
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_1▓
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_2љ
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_11/ActivityRegularizer/strided_slice│
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_11/ActivityRegularizer/Castо
$dense_11/ActivityRegularizer/truedivRealDiv5dense_11/ActivityRegularizer/PartitionedCall:output:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_11/ActivityRegularizer/truedivЭ
dropout_8/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_8_layer_call_and_return_conditional_losses_45362
dropout_8/PartitionedCallЎ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_12_4547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_45462"
 dense_12/StatefulPartitionedCallЭ
,dense_12/ActivityRegularizer/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_12_activity_regularizer_43802.
,dense_12/ActivityRegularizer/PartitionedCallА
"dense_12/ActivityRegularizer/ShapeShape)dense_12/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_12/ActivityRegularizer/Shape«
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_12/ActivityRegularizer/strided_slice/stack▓
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_1▓
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_2љ
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_12/ActivityRegularizer/strided_slice│
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_12/ActivityRegularizer/Castо
$dense_12/ActivityRegularizer/truedivRealDiv5dense_12/ActivityRegularizer/PartitionedCall:output:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_12/ActivityRegularizer/truedivЭ
dropout_9/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_9_layer_call_and_return_conditional_losses_45632
dropout_9/PartitionedCallЎ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_13_4574*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_45732"
 dense_13/StatefulPartitionedCallЭ
,dense_13/ActivityRegularizer/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_13_activity_regularizer_43932.
,dense_13/ActivityRegularizer/PartitionedCallА
"dense_13/ActivityRegularizer/ShapeShape)dense_13/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_13/ActivityRegularizer/Shape«
0dense_13/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_13/ActivityRegularizer/strided_slice/stack▓
2dense_13/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_1▓
2dense_13/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_2љ
*dense_13/ActivityRegularizer/strided_sliceStridedSlice+dense_13/ActivityRegularizer/Shape:output:09dense_13/ActivityRegularizer/strided_slice/stack:output:0;dense_13/ActivityRegularizer/strided_slice/stack_1:output:0;dense_13/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_13/ActivityRegularizer/strided_slice│
!dense_13/ActivityRegularizer/CastCast3dense_13/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_13/ActivityRegularizer/Castо
$dense_13/ActivityRegularizer/truedivRealDiv5dense_13/ActivityRegularizer/PartitionedCall:output:0%dense_13/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_13/ActivityRegularizer/truedivч
dropout_10/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_45902
dropout_10/PartitionedCallб
"net_output/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0net_output_4601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_net_output_layer_call_and_return_conditional_losses_46002$
"net_output/StatefulPartitionedCallђ
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *9
f4R2
0__inference_net_output_activity_regularizer_440620
.net_output/ActivityRegularizer/PartitionedCallД
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2&
$net_output/ActivityRegularizer/Shape▓
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2net_output/ActivityRegularizer/strided_slice/stackХ
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_1Х
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_2ю
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,net_output/ActivityRegularizer/strided_slice╣
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#net_output/ActivityRegularizer/Castя
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&net_output/ActivityRegularizer/truedivЂ
tf.math.subtract_2/Sub/yConst*
_output_shapes
:*
dtype0*
valueB*    2
tf.math.subtract_2/Sub/y╣
tf.math.subtract_2/SubSub+net_output/StatefulPartitionedCall:output:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:         2
tf.math.subtract_2/Subo
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.pow_1/Pow/yЎ
tf.math.pow_1/PowPowtf.math.subtract_2/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*'
_output_shapes
:         2
tf.math.pow_1/PowБ
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indicesх
tf.math.reduce_sum_1/SumSumtf.math.pow_1/Pow:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_1/Sumё
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constф
tf.math.reduce_mean_2/MeanMean!tf.math.reduce_sum_1/Sum:output:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/Meanб
 dense_14/StatefulPartitionedCallStatefulPartitionedCall+net_output/StatefulPartitionedCall:output:0dense_14_4629*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_46282"
 dense_14/StatefulPartitionedCallЭ
,dense_14/ActivityRegularizer/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_14_activity_regularizer_44192.
,dense_14/ActivityRegularizer/PartitionedCallА
"dense_14/ActivityRegularizer/ShapeShape)dense_14/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_14/ActivityRegularizer/Shape«
0dense_14/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_14/ActivityRegularizer/strided_slice/stack▓
2dense_14/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_1▓
2dense_14/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_2љ
*dense_14/ActivityRegularizer/strided_sliceStridedSlice+dense_14/ActivityRegularizer/Shape:output:09dense_14/ActivityRegularizer/strided_slice/stack:output:0;dense_14/ActivityRegularizer/strided_slice/stack_1:output:0;dense_14/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_14/ActivityRegularizer/strided_slice│
!dense_14/ActivityRegularizer/CastCast3dense_14/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_14/ActivityRegularizer/Castо
$dense_14/ActivityRegularizer/truedivRealDiv5dense_14/ActivityRegularizer/PartitionedCall:output:0%dense_14/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_14/ActivityRegularizer/truedivч
dropout_11/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_46452
dropout_11/PartitionedCallџ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_15_4656*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_46552"
 dense_15/StatefulPartitionedCallЭ
,dense_15/ActivityRegularizer/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_15_activity_regularizer_44322.
,dense_15/ActivityRegularizer/PartitionedCallА
"dense_15/ActivityRegularizer/ShapeShape)dense_15/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_15/ActivityRegularizer/Shape«
0dense_15/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_15/ActivityRegularizer/strided_slice/stack▓
2dense_15/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_1▓
2dense_15/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_2љ
*dense_15/ActivityRegularizer/strided_sliceStridedSlice+dense_15/ActivityRegularizer/Shape:output:09dense_15/ActivityRegularizer/strided_slice/stack:output:0;dense_15/ActivityRegularizer/strided_slice/stack_1:output:0;dense_15/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_15/ActivityRegularizer/strided_slice│
!dense_15/ActivityRegularizer/CastCast3dense_15/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_15/ActivityRegularizer/Castо
$dense_15/ActivityRegularizer/truedivRealDiv5dense_15/ActivityRegularizer/PartitionedCall:output:0%dense_15/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_15/ActivityRegularizer/truedivч
dropout_12/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_12_layer_call_and_return_conditional_losses_46722
dropout_12/PartitionedCallџ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_16_4683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_46822"
 dense_16/StatefulPartitionedCallЭ
,dense_16/ActivityRegularizer/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_16_activity_regularizer_44452.
,dense_16/ActivityRegularizer/PartitionedCallА
"dense_16/ActivityRegularizer/ShapeShape)dense_16/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape«
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack▓
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1▓
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2љ
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice│
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Castо
$dense_16/ActivityRegularizer/truedivRealDiv5dense_16/ActivityRegularizer/PartitionedCall:output:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_16/ActivityRegularizer/truedivч
dropout_13/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_46992
dropout_13/PartitionedCallџ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_17_4710*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_47092"
 dense_17/StatefulPartitionedCallЭ
,dense_17/ActivityRegularizer/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_17_activity_regularizer_44582.
,dense_17/ActivityRegularizer/PartitionedCallА
"dense_17/ActivityRegularizer/ShapeShape)dense_17/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_17/ActivityRegularizer/Shape«
0dense_17/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_17/ActivityRegularizer/strided_slice/stack▓
2dense_17/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_1▓
2dense_17/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_2љ
*dense_17/ActivityRegularizer/strided_sliceStridedSlice+dense_17/ActivityRegularizer/Shape:output:09dense_17/ActivityRegularizer/strided_slice/stack:output:0;dense_17/ActivityRegularizer/strided_slice/stack_1:output:0;dense_17/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_17/ActivityRegularizer/strided_slice│
!dense_17/ActivityRegularizer/CastCast3dense_17/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_17/ActivityRegularizer/Castо
$dense_17/ActivityRegularizer/truedivRealDiv5dense_17/ActivityRegularizer/PartitionedCall:output:0%dense_17/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_17/ActivityRegularizer/truedivч
dropout_14/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_14_layer_call_and_return_conditional_losses_47262
dropout_14/PartitionedCallџ
 dense_18/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0dense_18_4737*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_47362"
 dense_18/StatefulPartitionedCallЭ
,dense_18/ActivityRegularizer/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_18_activity_regularizer_44712.
,dense_18/ActivityRegularizer/PartitionedCallА
"dense_18/ActivityRegularizer/ShapeShape)dense_18/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_18/ActivityRegularizer/Shape«
0dense_18/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_18/ActivityRegularizer/strided_slice/stack▓
2dense_18/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_1▓
2dense_18/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_2љ
*dense_18/ActivityRegularizer/strided_sliceStridedSlice+dense_18/ActivityRegularizer/Shape:output:09dense_18/ActivityRegularizer/strided_slice/stack:output:0;dense_18/ActivityRegularizer/strided_slice/stack_1:output:0;dense_18/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_18/ActivityRegularizer/strided_slice│
!dense_18/ActivityRegularizer/CastCast3dense_18/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_18/ActivityRegularizer/Castо
$dense_18/ActivityRegularizer/truedivRealDiv5dense_18/ActivityRegularizer/PartitionedCall:output:0%dense_18/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_18/ActivityRegularizer/truedivч
dropout_15/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_15_layer_call_and_return_conditional_losses_47532
dropout_15/PartitionedCallџ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_19_4764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_47632"
 dense_19/StatefulPartitionedCallЭ
,dense_19/ActivityRegularizer/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_19_activity_regularizer_44842.
,dense_19/ActivityRegularizer/PartitionedCallА
"dense_19/ActivityRegularizer/ShapeShape)dense_19/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_19/ActivityRegularizer/Shape«
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_19/ActivityRegularizer/strided_slice/stack▓
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_1▓
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_2љ
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_19/ActivityRegularizer/strided_slice│
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_19/ActivityRegularizer/Castо
$dense_19/ActivityRegularizer/truedivRealDiv5dense_19/ActivityRegularizer/PartitionedCall:output:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_19/ActivityRegularizer/truedivю
tf.math.subtract_3/SubSub)dense_19/StatefulPartitionedCall:output:0inputs*
T0*'
_output_shapes
:         2
tf.math.subtract_3/Subі
tf.math.square_1/SquareSquaretf.math.subtract_3/Sub:z:0*
T0*'
_output_shapes
:         2
tf.math.square_1/SquareІ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constц
tf.math.reduce_mean_3/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean░
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_2/Mean:output:0#tf.math.reduce_mean_3/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_2/AddV2y
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 */Lь62
tf.__operators__.add_3/yФ
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_3/AddV2С
add_loss_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_add_loss_1_layer_call_and_return_conditional_losses_47872
add_loss_1/PartitionedCallx
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityv

Identity_1Identity(dense_10/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1v

Identity_2Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_2v

Identity_3Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3v

Identity_4Identity(dense_13/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_4x

Identity_5Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_5v

Identity_6Identity(dense_14/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_6v

Identity_7Identity(dense_15/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_7v

Identity_8Identity(dense_16/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_8v

Identity_9Identity(dense_17/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_9x
Identity_10Identity(dense_18/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_10x
Identity_11Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_11s
Identity_12Identity#add_loss_1/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 2
Identity_12Л
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─
{
'__inference_dense_12_layer_call_fn_6506

inputs
unknown:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_45462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┌
Г
D__inference_net_output_layer_call_and_return_conditional_losses_6874

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_12_layer_call_and_return_conditional_losses_6668

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_10_layer_call_and_return_conditional_losses_5108

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_16_layer_call_and_return_conditional_losses_4682

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
л

Ў
"__inference_signature_wrapper_5867
input_2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identityѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *(
f#R!
__inference__wrapped_model_43412
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
с
E
.__inference_dense_11_activity_regularizer_4367
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
зб
╣

A__inference_model_2_layer_call_and_return_conditional_losses_5432

inputs
dense_10_5273:
dense_11_5284:
dense_12_5296:
dense_13_5308:!
net_output_5320:
dense_14_5339:
dense_15_5351:
dense_16_5363:
dense_17_5375:
dense_18_5387:
dense_19_5399:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12ѕб dense_10/StatefulPartitionedCallб dense_11/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб dense_13/StatefulPartitionedCallб dense_14/StatefulPartitionedCallб dense_15/StatefulPartitionedCallб dense_16/StatefulPartitionedCallб dense_17/StatefulPartitionedCallб dense_18/StatefulPartitionedCallб dense_19/StatefulPartitionedCallб"dropout_10/StatefulPartitionedCallб"dropout_11/StatefulPartitionedCallб"dropout_12/StatefulPartitionedCallб"dropout_13/StatefulPartitionedCallб"dropout_14/StatefulPartitionedCallб"dropout_15/StatefulPartitionedCallб!dropout_8/StatefulPartitionedCallб!dropout_9/StatefulPartitionedCallб"net_output/StatefulPartitionedCall§
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_5273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_44992"
 dense_10/StatefulPartitionedCallЭ
,dense_10/ActivityRegularizer/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_10_activity_regularizer_43542.
,dense_10/ActivityRegularizer/PartitionedCallА
"dense_10/ActivityRegularizer/ShapeShape)dense_10/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_10/ActivityRegularizer/Shape«
0dense_10/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_10/ActivityRegularizer/strided_slice/stack▓
2dense_10/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_1▓
2dense_10/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_2љ
*dense_10/ActivityRegularizer/strided_sliceStridedSlice+dense_10/ActivityRegularizer/Shape:output:09dense_10/ActivityRegularizer/strided_slice/stack:output:0;dense_10/ActivityRegularizer/strided_slice/stack_1:output:0;dense_10/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_10/ActivityRegularizer/strided_slice│
!dense_10/ActivityRegularizer/CastCast3dense_10/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_10/ActivityRegularizer/Castо
$dense_10/ActivityRegularizer/truedivRealDiv5dense_10/ActivityRegularizer/PartitionedCall:output:0%dense_10/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_10/ActivityRegularizer/truedivа
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_5284*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_45192"
 dense_11/StatefulPartitionedCallЭ
,dense_11/ActivityRegularizer/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_11_activity_regularizer_43672.
,dense_11/ActivityRegularizer/PartitionedCallА
"dense_11/ActivityRegularizer/ShapeShape)dense_11/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_11/ActivityRegularizer/Shape«
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_11/ActivityRegularizer/strided_slice/stack▓
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_1▓
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_2љ
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_11/ActivityRegularizer/strided_slice│
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_11/ActivityRegularizer/Castо
$dense_11/ActivityRegularizer/truedivRealDiv5dense_11/ActivityRegularizer/PartitionedCall:output:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_11/ActivityRegularizer/truedivљ
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_8_layer_call_and_return_conditional_losses_51902#
!dropout_8/StatefulPartitionedCallА
 dense_12/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_12_5296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_45462"
 dense_12/StatefulPartitionedCallЭ
,dense_12/ActivityRegularizer/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_12_activity_regularizer_43802.
,dense_12/ActivityRegularizer/PartitionedCallА
"dense_12/ActivityRegularizer/ShapeShape)dense_12/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_12/ActivityRegularizer/Shape«
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_12/ActivityRegularizer/strided_slice/stack▓
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_1▓
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_2љ
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_12/ActivityRegularizer/strided_slice│
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_12/ActivityRegularizer/Castо
$dense_12/ActivityRegularizer/truedivRealDiv5dense_12/ActivityRegularizer/PartitionedCall:output:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_12/ActivityRegularizer/truediv┤
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_9_layer_call_and_return_conditional_losses_51492#
!dropout_9/StatefulPartitionedCallА
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_13_5308*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_45732"
 dense_13/StatefulPartitionedCallЭ
,dense_13/ActivityRegularizer/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_13_activity_regularizer_43932.
,dense_13/ActivityRegularizer/PartitionedCallА
"dense_13/ActivityRegularizer/ShapeShape)dense_13/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_13/ActivityRegularizer/Shape«
0dense_13/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_13/ActivityRegularizer/strided_slice/stack▓
2dense_13/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_1▓
2dense_13/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_2љ
*dense_13/ActivityRegularizer/strided_sliceStridedSlice+dense_13/ActivityRegularizer/Shape:output:09dense_13/ActivityRegularizer/strided_slice/stack:output:0;dense_13/ActivityRegularizer/strided_slice/stack_1:output:0;dense_13/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_13/ActivityRegularizer/strided_slice│
!dense_13/ActivityRegularizer/CastCast3dense_13/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_13/ActivityRegularizer/Castо
$dense_13/ActivityRegularizer/truedivRealDiv5dense_13/ActivityRegularizer/PartitionedCall:output:0%dense_13/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_13/ActivityRegularizer/truedivи
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_51082$
"dropout_10/StatefulPartitionedCallф
"net_output/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0net_output_5320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_net_output_layer_call_and_return_conditional_losses_46002$
"net_output/StatefulPartitionedCallђ
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *9
f4R2
0__inference_net_output_activity_regularizer_440620
.net_output/ActivityRegularizer/PartitionedCallД
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2&
$net_output/ActivityRegularizer/Shape▓
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2net_output/ActivityRegularizer/strided_slice/stackХ
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_1Х
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_2ю
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,net_output/ActivityRegularizer/strided_slice╣
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#net_output/ActivityRegularizer/Castя
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&net_output/ActivityRegularizer/truedivЂ
tf.math.subtract_2/Sub/yConst*
_output_shapes
:*
dtype0*
valueB*    2
tf.math.subtract_2/Sub/y╣
tf.math.subtract_2/SubSub+net_output/StatefulPartitionedCall:output:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:         2
tf.math.subtract_2/Subo
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.pow_1/Pow/yЎ
tf.math.pow_1/PowPowtf.math.subtract_2/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*'
_output_shapes
:         2
tf.math.pow_1/PowБ
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indicesх
tf.math.reduce_sum_1/SumSumtf.math.pow_1/Pow:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_1/Sumё
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constф
tf.math.reduce_mean_2/MeanMean!tf.math.reduce_sum_1/Sum:output:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/Meanб
 dense_14/StatefulPartitionedCallStatefulPartitionedCall+net_output/StatefulPartitionedCall:output:0dense_14_5339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_46282"
 dense_14/StatefulPartitionedCallЭ
,dense_14/ActivityRegularizer/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_14_activity_regularizer_44192.
,dense_14/ActivityRegularizer/PartitionedCallА
"dense_14/ActivityRegularizer/ShapeShape)dense_14/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_14/ActivityRegularizer/Shape«
0dense_14/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_14/ActivityRegularizer/strided_slice/stack▓
2dense_14/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_1▓
2dense_14/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_2љ
*dense_14/ActivityRegularizer/strided_sliceStridedSlice+dense_14/ActivityRegularizer/Shape:output:09dense_14/ActivityRegularizer/strided_slice/stack:output:0;dense_14/ActivityRegularizer/strided_slice/stack_1:output:0;dense_14/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_14/ActivityRegularizer/strided_slice│
!dense_14/ActivityRegularizer/CastCast3dense_14/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_14/ActivityRegularizer/Castо
$dense_14/ActivityRegularizer/truedivRealDiv5dense_14/ActivityRegularizer/PartitionedCall:output:0%dense_14/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_14/ActivityRegularizer/truedivИ
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_50492$
"dropout_11/StatefulPartitionedCallб
 dense_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_15_5351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_46552"
 dense_15/StatefulPartitionedCallЭ
,dense_15/ActivityRegularizer/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_15_activity_regularizer_44322.
,dense_15/ActivityRegularizer/PartitionedCallА
"dense_15/ActivityRegularizer/ShapeShape)dense_15/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_15/ActivityRegularizer/Shape«
0dense_15/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_15/ActivityRegularizer/strided_slice/stack▓
2dense_15/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_1▓
2dense_15/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_2љ
*dense_15/ActivityRegularizer/strided_sliceStridedSlice+dense_15/ActivityRegularizer/Shape:output:09dense_15/ActivityRegularizer/strided_slice/stack:output:0;dense_15/ActivityRegularizer/strided_slice/stack_1:output:0;dense_15/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_15/ActivityRegularizer/strided_slice│
!dense_15/ActivityRegularizer/CastCast3dense_15/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_15/ActivityRegularizer/Castо
$dense_15/ActivityRegularizer/truedivRealDiv5dense_15/ActivityRegularizer/PartitionedCall:output:0%dense_15/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_15/ActivityRegularizer/truedivИ
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_12_layer_call_and_return_conditional_losses_50082$
"dropout_12/StatefulPartitionedCallб
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_16_5363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_46822"
 dense_16/StatefulPartitionedCallЭ
,dense_16/ActivityRegularizer/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_16_activity_regularizer_44452.
,dense_16/ActivityRegularizer/PartitionedCallА
"dense_16/ActivityRegularizer/ShapeShape)dense_16/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape«
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack▓
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1▓
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2љ
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice│
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Castо
$dense_16/ActivityRegularizer/truedivRealDiv5dense_16/ActivityRegularizer/PartitionedCall:output:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_16/ActivityRegularizer/truedivИ
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_49672$
"dropout_13/StatefulPartitionedCallб
 dense_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_17_5375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_47092"
 dense_17/StatefulPartitionedCallЭ
,dense_17/ActivityRegularizer/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_17_activity_regularizer_44582.
,dense_17/ActivityRegularizer/PartitionedCallА
"dense_17/ActivityRegularizer/ShapeShape)dense_17/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_17/ActivityRegularizer/Shape«
0dense_17/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_17/ActivityRegularizer/strided_slice/stack▓
2dense_17/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_1▓
2dense_17/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_2љ
*dense_17/ActivityRegularizer/strided_sliceStridedSlice+dense_17/ActivityRegularizer/Shape:output:09dense_17/ActivityRegularizer/strided_slice/stack:output:0;dense_17/ActivityRegularizer/strided_slice/stack_1:output:0;dense_17/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_17/ActivityRegularizer/strided_slice│
!dense_17/ActivityRegularizer/CastCast3dense_17/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_17/ActivityRegularizer/Castо
$dense_17/ActivityRegularizer/truedivRealDiv5dense_17/ActivityRegularizer/PartitionedCall:output:0%dense_17/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_17/ActivityRegularizer/truedivИ
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0#^dropout_13/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_14_layer_call_and_return_conditional_losses_49262$
"dropout_14/StatefulPartitionedCallб
 dense_18/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0dense_18_5387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_47362"
 dense_18/StatefulPartitionedCallЭ
,dense_18/ActivityRegularizer/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_18_activity_regularizer_44712.
,dense_18/ActivityRegularizer/PartitionedCallА
"dense_18/ActivityRegularizer/ShapeShape)dense_18/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_18/ActivityRegularizer/Shape«
0dense_18/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_18/ActivityRegularizer/strided_slice/stack▓
2dense_18/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_1▓
2dense_18/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_2љ
*dense_18/ActivityRegularizer/strided_sliceStridedSlice+dense_18/ActivityRegularizer/Shape:output:09dense_18/ActivityRegularizer/strided_slice/stack:output:0;dense_18/ActivityRegularizer/strided_slice/stack_1:output:0;dense_18/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_18/ActivityRegularizer/strided_slice│
!dense_18/ActivityRegularizer/CastCast3dense_18/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_18/ActivityRegularizer/Castо
$dense_18/ActivityRegularizer/truedivRealDiv5dense_18/ActivityRegularizer/PartitionedCall:output:0%dense_18/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_18/ActivityRegularizer/truedivИ
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_15_layer_call_and_return_conditional_losses_48852$
"dropout_15/StatefulPartitionedCallб
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_19_5399*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_47632"
 dense_19/StatefulPartitionedCallЭ
,dense_19/ActivityRegularizer/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_19_activity_regularizer_44842.
,dense_19/ActivityRegularizer/PartitionedCallА
"dense_19/ActivityRegularizer/ShapeShape)dense_19/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_19/ActivityRegularizer/Shape«
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_19/ActivityRegularizer/strided_slice/stack▓
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_1▓
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_2љ
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_19/ActivityRegularizer/strided_slice│
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_19/ActivityRegularizer/Castо
$dense_19/ActivityRegularizer/truedivRealDiv5dense_19/ActivityRegularizer/PartitionedCall:output:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_19/ActivityRegularizer/truedivю
tf.math.subtract_3/SubSub)dense_19/StatefulPartitionedCall:output:0inputs*
T0*'
_output_shapes
:         2
tf.math.subtract_3/Subі
tf.math.square_1/SquareSquaretf.math.subtract_3/Sub:z:0*
T0*'
_output_shapes
:         2
tf.math.square_1/SquareІ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constц
tf.math.reduce_mean_3/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean░
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_2/Mean:output:0#tf.math.reduce_mean_3/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_2/AddV2y
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 */Lь62
tf.__operators__.add_3/yФ
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_3/AddV2С
add_loss_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_add_loss_1_layer_call_and_return_conditional_losses_47872
add_loss_1/PartitionedCallx
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityv

Identity_1Identity(dense_10/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1v

Identity_2Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_2v

Identity_3Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3v

Identity_4Identity(dense_13/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_4x

Identity_5Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_5v

Identity_6Identity(dense_14/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_6v

Identity_7Identity(dense_15/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_7v

Identity_8Identity(dense_16/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_8v

Identity_9Identity(dense_17/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_9x
Identity_10Identity(dense_18/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_10x
Identity_11Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_11s
Identity_12Identity#add_loss_1/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 2
Identity_12э
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_16_layer_call_and_return_conditional_losses_6898

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_14_layer_call_and_return_conditional_losses_6742

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
­
a
C__inference_dropout_8_layer_call_and_return_conditional_losses_4536

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
й
г
H__inference_net_output_layer_call_and_return_all_conditional_losses_6585

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_net_output_layer_call_and_return_conditional_losses_46002
StatefulPartitionedCallи
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *9
f4R2
0__inference_net_output_activity_regularizer_44062
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_14_layer_call_and_return_conditional_losses_6882

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_14_layer_call_and_return_conditional_losses_4926

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_10_layer_call_and_return_conditional_losses_4590

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┌
Ф
B__inference_dense_19_layer_call_and_return_conditional_losses_4763

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMula
SigmoidSigmoidMatMul:product:0*
T0*'
_output_shapes
:         2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_12_layer_call_and_return_conditional_losses_6656

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_14_layer_call_and_return_conditional_losses_4628

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_15_layer_call_and_return_conditional_losses_4885

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
ф
F__inference_dense_15_layer_call_and_return_all_conditional_losses_6644

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_46552
StatefulPartitionedCallх
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_15_activity_regularizer_44322
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_13_layer_call_and_return_conditional_losses_4967

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─
{
'__inference_dense_16_layer_call_fn_6694

inputs
unknown:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_46822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_10_layer_call_and_return_conditional_losses_4499

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
ф
F__inference_dense_13_layer_call_and_return_all_conditional_losses_6542

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_45732
StatefulPartitionedCallх
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_13_activity_regularizer_43932
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_15_layer_call_and_return_conditional_losses_4753

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
¤
p
D__inference_add_loss_1_layer_call_and_return_conditional_losses_6828

inputs
identity

identity_1I
IdentityIdentityinputs*
T0*
_output_shapes
: 2

IdentityM

Identity_1Identityinputs*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
п
Ф
B__inference_dense_10_layer_call_and_return_conditional_losses_6842

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
ф
F__inference_dense_10_layer_call_and_return_all_conditional_losses_6440

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_44992
StatefulPartitionedCallх
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_10_activity_regularizer_43542
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
║
D
(__inference_dropout_8_layer_call_fn_6485

inputs
identity┴
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_8_layer_call_and_return_conditional_losses_45362
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
║
D
(__inference_dropout_9_layer_call_fn_6528

inputs
identity┴
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_9_layer_call_and_return_conditional_losses_45632
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
џ
Ю
&__inference_model_2_layer_call_fn_5508
input_2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identityѕбStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         : : : : : : : : : : : : *-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_54322
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
п
Ф
B__inference_dense_11_layer_call_and_return_conditional_losses_4519

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
­
a
C__inference_dropout_9_layer_call_and_return_conditional_losses_6511

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ъ
b
)__inference_dropout_10_layer_call_fn_6576

inputs
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_51082
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_13_layer_call_and_return_conditional_losses_6711

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ЦК
у	
__inference__wrapped_model_4341
input_2A
/model_2_dense_10_matmul_readvariableop_resource:A
/model_2_dense_11_matmul_readvariableop_resource:A
/model_2_dense_12_matmul_readvariableop_resource:A
/model_2_dense_13_matmul_readvariableop_resource:C
1model_2_net_output_matmul_readvariableop_resource:A
/model_2_dense_14_matmul_readvariableop_resource:A
/model_2_dense_15_matmul_readvariableop_resource:A
/model_2_dense_16_matmul_readvariableop_resource:A
/model_2_dense_17_matmul_readvariableop_resource:A
/model_2_dense_18_matmul_readvariableop_resource:A
/model_2_dense_19_matmul_readvariableop_resource:
identityѕб&model_2/dense_10/MatMul/ReadVariableOpб&model_2/dense_11/MatMul/ReadVariableOpб&model_2/dense_12/MatMul/ReadVariableOpб&model_2/dense_13/MatMul/ReadVariableOpб&model_2/dense_14/MatMul/ReadVariableOpб&model_2/dense_15/MatMul/ReadVariableOpб&model_2/dense_16/MatMul/ReadVariableOpб&model_2/dense_17/MatMul/ReadVariableOpб&model_2/dense_18/MatMul/ReadVariableOpб&model_2/dense_19/MatMul/ReadVariableOpб(model_2/net_output/MatMul/ReadVariableOp└
&model_2/dense_10/MatMul/ReadVariableOpReadVariableOp/model_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_10/MatMul/ReadVariableOpД
model_2/dense_10/MatMulMatMulinput_2.model_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/dense_10/MatMulІ
model_2/dense_10/ReluRelu!model_2/dense_10/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/dense_10/Relu╗
+model_2/dense_10/ActivityRegularizer/SquareSquare#model_2/dense_10/Relu:activations:0*
T0*'
_output_shapes
:         2-
+model_2/dense_10/ActivityRegularizer/SquareЕ
*model_2/dense_10/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model_2/dense_10/ActivityRegularizer/ConstР
(model_2/dense_10/ActivityRegularizer/SumSum/model_2/dense_10/ActivityRegularizer/Square:y:03model_2/dense_10/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_10/ActivityRegularizer/SumЮ
*model_2/dense_10/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2,
*model_2/dense_10/ActivityRegularizer/mul/xС
(model_2/dense_10/ActivityRegularizer/mulMul3model_2/dense_10/ActivityRegularizer/mul/x:output:01model_2/dense_10/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_10/ActivityRegularizer/mulФ
*model_2/dense_10/ActivityRegularizer/ShapeShape#model_2/dense_10/Relu:activations:0*
T0*
_output_shapes
:2,
*model_2/dense_10/ActivityRegularizer/ShapeЙ
8model_2/dense_10/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_2/dense_10/ActivityRegularizer/strided_slice/stack┬
:model_2/dense_10/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_10/ActivityRegularizer/strided_slice/stack_1┬
:model_2/dense_10/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_10/ActivityRegularizer/strided_slice/stack_2└
2model_2/dense_10/ActivityRegularizer/strided_sliceStridedSlice3model_2/dense_10/ActivityRegularizer/Shape:output:0Amodel_2/dense_10/ActivityRegularizer/strided_slice/stack:output:0Cmodel_2/dense_10/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_2/dense_10/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_2/dense_10/ActivityRegularizer/strided_slice╦
)model_2/dense_10/ActivityRegularizer/CastCast;model_2/dense_10/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_2/dense_10/ActivityRegularizer/Castт
,model_2/dense_10/ActivityRegularizer/truedivRealDiv,model_2/dense_10/ActivityRegularizer/mul:z:0-model_2/dense_10/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_2/dense_10/ActivityRegularizer/truediv└
&model_2/dense_11/MatMul/ReadVariableOpReadVariableOp/model_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_11/MatMul/ReadVariableOp├
model_2/dense_11/MatMulMatMul#model_2/dense_10/Relu:activations:0.model_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/dense_11/MatMulІ
model_2/dense_11/ReluRelu!model_2/dense_11/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/dense_11/Relu╗
+model_2/dense_11/ActivityRegularizer/SquareSquare#model_2/dense_11/Relu:activations:0*
T0*'
_output_shapes
:         2-
+model_2/dense_11/ActivityRegularizer/SquareЕ
*model_2/dense_11/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model_2/dense_11/ActivityRegularizer/ConstР
(model_2/dense_11/ActivityRegularizer/SumSum/model_2/dense_11/ActivityRegularizer/Square:y:03model_2/dense_11/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_11/ActivityRegularizer/SumЮ
*model_2/dense_11/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2,
*model_2/dense_11/ActivityRegularizer/mul/xС
(model_2/dense_11/ActivityRegularizer/mulMul3model_2/dense_11/ActivityRegularizer/mul/x:output:01model_2/dense_11/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_11/ActivityRegularizer/mulФ
*model_2/dense_11/ActivityRegularizer/ShapeShape#model_2/dense_11/Relu:activations:0*
T0*
_output_shapes
:2,
*model_2/dense_11/ActivityRegularizer/ShapeЙ
8model_2/dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_2/dense_11/ActivityRegularizer/strided_slice/stack┬
:model_2/dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_11/ActivityRegularizer/strided_slice/stack_1┬
:model_2/dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_11/ActivityRegularizer/strided_slice/stack_2└
2model_2/dense_11/ActivityRegularizer/strided_sliceStridedSlice3model_2/dense_11/ActivityRegularizer/Shape:output:0Amodel_2/dense_11/ActivityRegularizer/strided_slice/stack:output:0Cmodel_2/dense_11/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_2/dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_2/dense_11/ActivityRegularizer/strided_slice╦
)model_2/dense_11/ActivityRegularizer/CastCast;model_2/dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_2/dense_11/ActivityRegularizer/Castт
,model_2/dense_11/ActivityRegularizer/truedivRealDiv,model_2/dense_11/ActivityRegularizer/mul:z:0-model_2/dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_2/dense_11/ActivityRegularizer/truedivЏ
model_2/dropout_8/IdentityIdentity#model_2/dense_11/Relu:activations:0*
T0*'
_output_shapes
:         2
model_2/dropout_8/Identity└
&model_2/dense_12/MatMul/ReadVariableOpReadVariableOp/model_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_12/MatMul/ReadVariableOp├
model_2/dense_12/MatMulMatMul#model_2/dropout_8/Identity:output:0.model_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/dense_12/MatMulІ
model_2/dense_12/ReluRelu!model_2/dense_12/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/dense_12/Relu╗
+model_2/dense_12/ActivityRegularizer/SquareSquare#model_2/dense_12/Relu:activations:0*
T0*'
_output_shapes
:         2-
+model_2/dense_12/ActivityRegularizer/SquareЕ
*model_2/dense_12/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model_2/dense_12/ActivityRegularizer/ConstР
(model_2/dense_12/ActivityRegularizer/SumSum/model_2/dense_12/ActivityRegularizer/Square:y:03model_2/dense_12/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_12/ActivityRegularizer/SumЮ
*model_2/dense_12/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2,
*model_2/dense_12/ActivityRegularizer/mul/xС
(model_2/dense_12/ActivityRegularizer/mulMul3model_2/dense_12/ActivityRegularizer/mul/x:output:01model_2/dense_12/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_12/ActivityRegularizer/mulФ
*model_2/dense_12/ActivityRegularizer/ShapeShape#model_2/dense_12/Relu:activations:0*
T0*
_output_shapes
:2,
*model_2/dense_12/ActivityRegularizer/ShapeЙ
8model_2/dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_2/dense_12/ActivityRegularizer/strided_slice/stack┬
:model_2/dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_12/ActivityRegularizer/strided_slice/stack_1┬
:model_2/dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_12/ActivityRegularizer/strided_slice/stack_2└
2model_2/dense_12/ActivityRegularizer/strided_sliceStridedSlice3model_2/dense_12/ActivityRegularizer/Shape:output:0Amodel_2/dense_12/ActivityRegularizer/strided_slice/stack:output:0Cmodel_2/dense_12/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_2/dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_2/dense_12/ActivityRegularizer/strided_slice╦
)model_2/dense_12/ActivityRegularizer/CastCast;model_2/dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_2/dense_12/ActivityRegularizer/Castт
,model_2/dense_12/ActivityRegularizer/truedivRealDiv,model_2/dense_12/ActivityRegularizer/mul:z:0-model_2/dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_2/dense_12/ActivityRegularizer/truedivЏ
model_2/dropout_9/IdentityIdentity#model_2/dense_12/Relu:activations:0*
T0*'
_output_shapes
:         2
model_2/dropout_9/Identity└
&model_2/dense_13/MatMul/ReadVariableOpReadVariableOp/model_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_13/MatMul/ReadVariableOp├
model_2/dense_13/MatMulMatMul#model_2/dropout_9/Identity:output:0.model_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/dense_13/MatMulІ
model_2/dense_13/ReluRelu!model_2/dense_13/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/dense_13/Relu╗
+model_2/dense_13/ActivityRegularizer/SquareSquare#model_2/dense_13/Relu:activations:0*
T0*'
_output_shapes
:         2-
+model_2/dense_13/ActivityRegularizer/SquareЕ
*model_2/dense_13/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model_2/dense_13/ActivityRegularizer/ConstР
(model_2/dense_13/ActivityRegularizer/SumSum/model_2/dense_13/ActivityRegularizer/Square:y:03model_2/dense_13/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_13/ActivityRegularizer/SumЮ
*model_2/dense_13/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2,
*model_2/dense_13/ActivityRegularizer/mul/xС
(model_2/dense_13/ActivityRegularizer/mulMul3model_2/dense_13/ActivityRegularizer/mul/x:output:01model_2/dense_13/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_13/ActivityRegularizer/mulФ
*model_2/dense_13/ActivityRegularizer/ShapeShape#model_2/dense_13/Relu:activations:0*
T0*
_output_shapes
:2,
*model_2/dense_13/ActivityRegularizer/ShapeЙ
8model_2/dense_13/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_2/dense_13/ActivityRegularizer/strided_slice/stack┬
:model_2/dense_13/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_13/ActivityRegularizer/strided_slice/stack_1┬
:model_2/dense_13/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_13/ActivityRegularizer/strided_slice/stack_2└
2model_2/dense_13/ActivityRegularizer/strided_sliceStridedSlice3model_2/dense_13/ActivityRegularizer/Shape:output:0Amodel_2/dense_13/ActivityRegularizer/strided_slice/stack:output:0Cmodel_2/dense_13/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_2/dense_13/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_2/dense_13/ActivityRegularizer/strided_slice╦
)model_2/dense_13/ActivityRegularizer/CastCast;model_2/dense_13/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_2/dense_13/ActivityRegularizer/Castт
,model_2/dense_13/ActivityRegularizer/truedivRealDiv,model_2/dense_13/ActivityRegularizer/mul:z:0-model_2/dense_13/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_2/dense_13/ActivityRegularizer/truedivЮ
model_2/dropout_10/IdentityIdentity#model_2/dense_13/Relu:activations:0*
T0*'
_output_shapes
:         2
model_2/dropout_10/Identityк
(model_2/net_output/MatMul/ReadVariableOpReadVariableOp1model_2_net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(model_2/net_output/MatMul/ReadVariableOp╩
model_2/net_output/MatMulMatMul$model_2/dropout_10/Identity:output:00model_2/net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/net_output/MatMulЉ
model_2/net_output/ReluRelu#model_2/net_output/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/net_output/Relu┴
-model_2/net_output/ActivityRegularizer/SquareSquare%model_2/net_output/Relu:activations:0*
T0*'
_output_shapes
:         2/
-model_2/net_output/ActivityRegularizer/SquareГ
,model_2/net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_2/net_output/ActivityRegularizer/ConstЖ
*model_2/net_output/ActivityRegularizer/SumSum1model_2/net_output/ActivityRegularizer/Square:y:05model_2/net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2,
*model_2/net_output/ActivityRegularizer/SumА
,model_2/net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2.
,model_2/net_output/ActivityRegularizer/mul/xВ
*model_2/net_output/ActivityRegularizer/mulMul5model_2/net_output/ActivityRegularizer/mul/x:output:03model_2/net_output/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*model_2/net_output/ActivityRegularizer/mul▒
,model_2/net_output/ActivityRegularizer/ShapeShape%model_2/net_output/Relu:activations:0*
T0*
_output_shapes
:2.
,model_2/net_output/ActivityRegularizer/Shape┬
:model_2/net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:model_2/net_output/ActivityRegularizer/strided_slice/stackк
<model_2/net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_2/net_output/ActivityRegularizer/strided_slice/stack_1к
<model_2/net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_2/net_output/ActivityRegularizer/strided_slice/stack_2╠
4model_2/net_output/ActivityRegularizer/strided_sliceStridedSlice5model_2/net_output/ActivityRegularizer/Shape:output:0Cmodel_2/net_output/ActivityRegularizer/strided_slice/stack:output:0Emodel_2/net_output/ActivityRegularizer/strided_slice/stack_1:output:0Emodel_2/net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_2/net_output/ActivityRegularizer/strided_sliceЛ
+model_2/net_output/ActivityRegularizer/CastCast=model_2/net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+model_2/net_output/ActivityRegularizer/Castь
.model_2/net_output/ActivityRegularizer/truedivRealDiv.model_2/net_output/ActivityRegularizer/mul:z:0/model_2/net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 20
.model_2/net_output/ActivityRegularizer/truedivЉ
 model_2/tf.math.subtract_2/Sub/yConst*
_output_shapes
:*
dtype0*
valueB*    2"
 model_2/tf.math.subtract_2/Sub/y╦
model_2/tf.math.subtract_2/SubSub%model_2/net_output/Relu:activations:0)model_2/tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:         2 
model_2/tf.math.subtract_2/Sub
model_2/tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_2/tf.math.pow_1/Pow/y╣
model_2/tf.math.pow_1/PowPow"model_2/tf.math.subtract_2/Sub:z:0$model_2/tf.math.pow_1/Pow/y:output:0*
T0*'
_output_shapes
:         2
model_2/tf.math.pow_1/Pow│
2model_2/tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         24
2model_2/tf.math.reduce_sum_1/Sum/reduction_indicesН
 model_2/tf.math.reduce_sum_1/SumSummodel_2/tf.math.pow_1/Pow:z:0;model_2/tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2"
 model_2/tf.math.reduce_sum_1/Sumћ
#model_2/tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_2/tf.math.reduce_mean_2/Const╩
"model_2/tf.math.reduce_mean_2/MeanMean)model_2/tf.math.reduce_sum_1/Sum:output:0,model_2/tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2$
"model_2/tf.math.reduce_mean_2/Mean└
&model_2/dense_14/MatMul/ReadVariableOpReadVariableOp/model_2_dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_14/MatMul/ReadVariableOp┼
model_2/dense_14/MatMulMatMul%model_2/net_output/Relu:activations:0.model_2/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/dense_14/MatMulІ
model_2/dense_14/ReluRelu!model_2/dense_14/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/dense_14/Relu╗
+model_2/dense_14/ActivityRegularizer/SquareSquare#model_2/dense_14/Relu:activations:0*
T0*'
_output_shapes
:         2-
+model_2/dense_14/ActivityRegularizer/SquareЕ
*model_2/dense_14/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model_2/dense_14/ActivityRegularizer/ConstР
(model_2/dense_14/ActivityRegularizer/SumSum/model_2/dense_14/ActivityRegularizer/Square:y:03model_2/dense_14/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_14/ActivityRegularizer/SumЮ
*model_2/dense_14/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2,
*model_2/dense_14/ActivityRegularizer/mul/xС
(model_2/dense_14/ActivityRegularizer/mulMul3model_2/dense_14/ActivityRegularizer/mul/x:output:01model_2/dense_14/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_14/ActivityRegularizer/mulФ
*model_2/dense_14/ActivityRegularizer/ShapeShape#model_2/dense_14/Relu:activations:0*
T0*
_output_shapes
:2,
*model_2/dense_14/ActivityRegularizer/ShapeЙ
8model_2/dense_14/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_2/dense_14/ActivityRegularizer/strided_slice/stack┬
:model_2/dense_14/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_14/ActivityRegularizer/strided_slice/stack_1┬
:model_2/dense_14/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_14/ActivityRegularizer/strided_slice/stack_2└
2model_2/dense_14/ActivityRegularizer/strided_sliceStridedSlice3model_2/dense_14/ActivityRegularizer/Shape:output:0Amodel_2/dense_14/ActivityRegularizer/strided_slice/stack:output:0Cmodel_2/dense_14/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_2/dense_14/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_2/dense_14/ActivityRegularizer/strided_slice╦
)model_2/dense_14/ActivityRegularizer/CastCast;model_2/dense_14/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_2/dense_14/ActivityRegularizer/Castт
,model_2/dense_14/ActivityRegularizer/truedivRealDiv,model_2/dense_14/ActivityRegularizer/mul:z:0-model_2/dense_14/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_2/dense_14/ActivityRegularizer/truedivЮ
model_2/dropout_11/IdentityIdentity#model_2/dense_14/Relu:activations:0*
T0*'
_output_shapes
:         2
model_2/dropout_11/Identity└
&model_2/dense_15/MatMul/ReadVariableOpReadVariableOp/model_2_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_15/MatMul/ReadVariableOp─
model_2/dense_15/MatMulMatMul$model_2/dropout_11/Identity:output:0.model_2/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/dense_15/MatMulІ
model_2/dense_15/ReluRelu!model_2/dense_15/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/dense_15/Relu╗
+model_2/dense_15/ActivityRegularizer/SquareSquare#model_2/dense_15/Relu:activations:0*
T0*'
_output_shapes
:         2-
+model_2/dense_15/ActivityRegularizer/SquareЕ
*model_2/dense_15/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model_2/dense_15/ActivityRegularizer/ConstР
(model_2/dense_15/ActivityRegularizer/SumSum/model_2/dense_15/ActivityRegularizer/Square:y:03model_2/dense_15/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_15/ActivityRegularizer/SumЮ
*model_2/dense_15/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2,
*model_2/dense_15/ActivityRegularizer/mul/xС
(model_2/dense_15/ActivityRegularizer/mulMul3model_2/dense_15/ActivityRegularizer/mul/x:output:01model_2/dense_15/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_15/ActivityRegularizer/mulФ
*model_2/dense_15/ActivityRegularizer/ShapeShape#model_2/dense_15/Relu:activations:0*
T0*
_output_shapes
:2,
*model_2/dense_15/ActivityRegularizer/ShapeЙ
8model_2/dense_15/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_2/dense_15/ActivityRegularizer/strided_slice/stack┬
:model_2/dense_15/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_15/ActivityRegularizer/strided_slice/stack_1┬
:model_2/dense_15/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_15/ActivityRegularizer/strided_slice/stack_2└
2model_2/dense_15/ActivityRegularizer/strided_sliceStridedSlice3model_2/dense_15/ActivityRegularizer/Shape:output:0Amodel_2/dense_15/ActivityRegularizer/strided_slice/stack:output:0Cmodel_2/dense_15/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_2/dense_15/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_2/dense_15/ActivityRegularizer/strided_slice╦
)model_2/dense_15/ActivityRegularizer/CastCast;model_2/dense_15/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_2/dense_15/ActivityRegularizer/Castт
,model_2/dense_15/ActivityRegularizer/truedivRealDiv,model_2/dense_15/ActivityRegularizer/mul:z:0-model_2/dense_15/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_2/dense_15/ActivityRegularizer/truedivЮ
model_2/dropout_12/IdentityIdentity#model_2/dense_15/Relu:activations:0*
T0*'
_output_shapes
:         2
model_2/dropout_12/Identity└
&model_2/dense_16/MatMul/ReadVariableOpReadVariableOp/model_2_dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_16/MatMul/ReadVariableOp─
model_2/dense_16/MatMulMatMul$model_2/dropout_12/Identity:output:0.model_2/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/dense_16/MatMulІ
model_2/dense_16/ReluRelu!model_2/dense_16/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/dense_16/Relu╗
+model_2/dense_16/ActivityRegularizer/SquareSquare#model_2/dense_16/Relu:activations:0*
T0*'
_output_shapes
:         2-
+model_2/dense_16/ActivityRegularizer/SquareЕ
*model_2/dense_16/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model_2/dense_16/ActivityRegularizer/ConstР
(model_2/dense_16/ActivityRegularizer/SumSum/model_2/dense_16/ActivityRegularizer/Square:y:03model_2/dense_16/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_16/ActivityRegularizer/SumЮ
*model_2/dense_16/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2,
*model_2/dense_16/ActivityRegularizer/mul/xС
(model_2/dense_16/ActivityRegularizer/mulMul3model_2/dense_16/ActivityRegularizer/mul/x:output:01model_2/dense_16/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_16/ActivityRegularizer/mulФ
*model_2/dense_16/ActivityRegularizer/ShapeShape#model_2/dense_16/Relu:activations:0*
T0*
_output_shapes
:2,
*model_2/dense_16/ActivityRegularizer/ShapeЙ
8model_2/dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_2/dense_16/ActivityRegularizer/strided_slice/stack┬
:model_2/dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_16/ActivityRegularizer/strided_slice/stack_1┬
:model_2/dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_16/ActivityRegularizer/strided_slice/stack_2└
2model_2/dense_16/ActivityRegularizer/strided_sliceStridedSlice3model_2/dense_16/ActivityRegularizer/Shape:output:0Amodel_2/dense_16/ActivityRegularizer/strided_slice/stack:output:0Cmodel_2/dense_16/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_2/dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_2/dense_16/ActivityRegularizer/strided_slice╦
)model_2/dense_16/ActivityRegularizer/CastCast;model_2/dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_2/dense_16/ActivityRegularizer/Castт
,model_2/dense_16/ActivityRegularizer/truedivRealDiv,model_2/dense_16/ActivityRegularizer/mul:z:0-model_2/dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_2/dense_16/ActivityRegularizer/truedivЮ
model_2/dropout_13/IdentityIdentity#model_2/dense_16/Relu:activations:0*
T0*'
_output_shapes
:         2
model_2/dropout_13/Identity└
&model_2/dense_17/MatMul/ReadVariableOpReadVariableOp/model_2_dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_17/MatMul/ReadVariableOp─
model_2/dense_17/MatMulMatMul$model_2/dropout_13/Identity:output:0.model_2/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/dense_17/MatMulІ
model_2/dense_17/ReluRelu!model_2/dense_17/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/dense_17/Relu╗
+model_2/dense_17/ActivityRegularizer/SquareSquare#model_2/dense_17/Relu:activations:0*
T0*'
_output_shapes
:         2-
+model_2/dense_17/ActivityRegularizer/SquareЕ
*model_2/dense_17/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model_2/dense_17/ActivityRegularizer/ConstР
(model_2/dense_17/ActivityRegularizer/SumSum/model_2/dense_17/ActivityRegularizer/Square:y:03model_2/dense_17/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_17/ActivityRegularizer/SumЮ
*model_2/dense_17/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2,
*model_2/dense_17/ActivityRegularizer/mul/xС
(model_2/dense_17/ActivityRegularizer/mulMul3model_2/dense_17/ActivityRegularizer/mul/x:output:01model_2/dense_17/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_17/ActivityRegularizer/mulФ
*model_2/dense_17/ActivityRegularizer/ShapeShape#model_2/dense_17/Relu:activations:0*
T0*
_output_shapes
:2,
*model_2/dense_17/ActivityRegularizer/ShapeЙ
8model_2/dense_17/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_2/dense_17/ActivityRegularizer/strided_slice/stack┬
:model_2/dense_17/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_17/ActivityRegularizer/strided_slice/stack_1┬
:model_2/dense_17/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_17/ActivityRegularizer/strided_slice/stack_2└
2model_2/dense_17/ActivityRegularizer/strided_sliceStridedSlice3model_2/dense_17/ActivityRegularizer/Shape:output:0Amodel_2/dense_17/ActivityRegularizer/strided_slice/stack:output:0Cmodel_2/dense_17/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_2/dense_17/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_2/dense_17/ActivityRegularizer/strided_slice╦
)model_2/dense_17/ActivityRegularizer/CastCast;model_2/dense_17/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_2/dense_17/ActivityRegularizer/Castт
,model_2/dense_17/ActivityRegularizer/truedivRealDiv,model_2/dense_17/ActivityRegularizer/mul:z:0-model_2/dense_17/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_2/dense_17/ActivityRegularizer/truedivЮ
model_2/dropout_14/IdentityIdentity#model_2/dense_17/Relu:activations:0*
T0*'
_output_shapes
:         2
model_2/dropout_14/Identity└
&model_2/dense_18/MatMul/ReadVariableOpReadVariableOp/model_2_dense_18_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_18/MatMul/ReadVariableOp─
model_2/dense_18/MatMulMatMul$model_2/dropout_14/Identity:output:0.model_2/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/dense_18/MatMulІ
model_2/dense_18/ReluRelu!model_2/dense_18/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/dense_18/Relu╗
+model_2/dense_18/ActivityRegularizer/SquareSquare#model_2/dense_18/Relu:activations:0*
T0*'
_output_shapes
:         2-
+model_2/dense_18/ActivityRegularizer/SquareЕ
*model_2/dense_18/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model_2/dense_18/ActivityRegularizer/ConstР
(model_2/dense_18/ActivityRegularizer/SumSum/model_2/dense_18/ActivityRegularizer/Square:y:03model_2/dense_18/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_18/ActivityRegularizer/SumЮ
*model_2/dense_18/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2,
*model_2/dense_18/ActivityRegularizer/mul/xС
(model_2/dense_18/ActivityRegularizer/mulMul3model_2/dense_18/ActivityRegularizer/mul/x:output:01model_2/dense_18/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_18/ActivityRegularizer/mulФ
*model_2/dense_18/ActivityRegularizer/ShapeShape#model_2/dense_18/Relu:activations:0*
T0*
_output_shapes
:2,
*model_2/dense_18/ActivityRegularizer/ShapeЙ
8model_2/dense_18/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_2/dense_18/ActivityRegularizer/strided_slice/stack┬
:model_2/dense_18/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_18/ActivityRegularizer/strided_slice/stack_1┬
:model_2/dense_18/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_18/ActivityRegularizer/strided_slice/stack_2└
2model_2/dense_18/ActivityRegularizer/strided_sliceStridedSlice3model_2/dense_18/ActivityRegularizer/Shape:output:0Amodel_2/dense_18/ActivityRegularizer/strided_slice/stack:output:0Cmodel_2/dense_18/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_2/dense_18/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_2/dense_18/ActivityRegularizer/strided_slice╦
)model_2/dense_18/ActivityRegularizer/CastCast;model_2/dense_18/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_2/dense_18/ActivityRegularizer/Castт
,model_2/dense_18/ActivityRegularizer/truedivRealDiv,model_2/dense_18/ActivityRegularizer/mul:z:0-model_2/dense_18/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_2/dense_18/ActivityRegularizer/truedivЮ
model_2/dropout_15/IdentityIdentity#model_2/dense_18/Relu:activations:0*
T0*'
_output_shapes
:         2
model_2/dropout_15/Identity└
&model_2/dense_19/MatMul/ReadVariableOpReadVariableOp/model_2_dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_2/dense_19/MatMul/ReadVariableOp─
model_2/dense_19/MatMulMatMul$model_2/dropout_15/Identity:output:0.model_2/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/dense_19/MatMulћ
model_2/dense_19/SigmoidSigmoid!model_2/dense_19/MatMul:product:0*
T0*'
_output_shapes
:         2
model_2/dense_19/Sigmoid┤
+model_2/dense_19/ActivityRegularizer/SquareSquaremodel_2/dense_19/Sigmoid:y:0*
T0*'
_output_shapes
:         2-
+model_2/dense_19/ActivityRegularizer/SquareЕ
*model_2/dense_19/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*model_2/dense_19/ActivityRegularizer/ConstР
(model_2/dense_19/ActivityRegularizer/SumSum/model_2/dense_19/ActivityRegularizer/Square:y:03model_2/dense_19/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_19/ActivityRegularizer/SumЮ
*model_2/dense_19/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2,
*model_2/dense_19/ActivityRegularizer/mul/xС
(model_2/dense_19/ActivityRegularizer/mulMul3model_2/dense_19/ActivityRegularizer/mul/x:output:01model_2/dense_19/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_2/dense_19/ActivityRegularizer/mulц
*model_2/dense_19/ActivityRegularizer/ShapeShapemodel_2/dense_19/Sigmoid:y:0*
T0*
_output_shapes
:2,
*model_2/dense_19/ActivityRegularizer/ShapeЙ
8model_2/dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_2/dense_19/ActivityRegularizer/strided_slice/stack┬
:model_2/dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_19/ActivityRegularizer/strided_slice/stack_1┬
:model_2/dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_2/dense_19/ActivityRegularizer/strided_slice/stack_2└
2model_2/dense_19/ActivityRegularizer/strided_sliceStridedSlice3model_2/dense_19/ActivityRegularizer/Shape:output:0Amodel_2/dense_19/ActivityRegularizer/strided_slice/stack:output:0Cmodel_2/dense_19/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_2/dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_2/dense_19/ActivityRegularizer/strided_slice╦
)model_2/dense_19/ActivityRegularizer/CastCast;model_2/dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_2/dense_19/ActivityRegularizer/Castт
,model_2/dense_19/ActivityRegularizer/truedivRealDiv,model_2/dense_19/ActivityRegularizer/mul:z:0-model_2/dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_2/dense_19/ActivityRegularizer/truedivа
model_2/tf.math.subtract_3/SubSubmodel_2/dense_19/Sigmoid:y:0input_2*
T0*'
_output_shapes
:         2 
model_2/tf.math.subtract_3/Subб
model_2/tf.math.square_1/SquareSquare"model_2/tf.math.subtract_3/Sub:z:0*
T0*'
_output_shapes
:         2!
model_2/tf.math.square_1/SquareЏ
#model_2/tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#model_2/tf.math.reduce_mean_3/Const─
"model_2/tf.math.reduce_mean_3/MeanMean#model_2/tf.math.square_1/Square:y:0,model_2/tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2$
"model_2/tf.math.reduce_mean_3/Meanл
$model_2/tf.__operators__.add_2/AddV2AddV2+model_2/tf.math.reduce_mean_2/Mean:output:0+model_2/tf.math.reduce_mean_3/Mean:output:0*
T0*
_output_shapes
: 2&
$model_2/tf.__operators__.add_2/AddV2Ѕ
 model_2/tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 */Lь62"
 model_2/tf.__operators__.add_3/y╦
$model_2/tf.__operators__.add_3/AddV2AddV2(model_2/tf.__operators__.add_2/AddV2:z:0)model_2/tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: 2&
$model_2/tf.__operators__.add_3/AddV2ђ
IdentityIdentity)model_2/tf.math.reduce_sum_1/Sum:output:0^NoOp*
T0*#
_output_shapes
:         2

IdentityЊ
NoOpNoOp'^model_2/dense_10/MatMul/ReadVariableOp'^model_2/dense_11/MatMul/ReadVariableOp'^model_2/dense_12/MatMul/ReadVariableOp'^model_2/dense_13/MatMul/ReadVariableOp'^model_2/dense_14/MatMul/ReadVariableOp'^model_2/dense_15/MatMul/ReadVariableOp'^model_2/dense_16/MatMul/ReadVariableOp'^model_2/dense_17/MatMul/ReadVariableOp'^model_2/dense_18/MatMul/ReadVariableOp'^model_2/dense_19/MatMul/ReadVariableOp)^model_2/net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2P
&model_2/dense_10/MatMul/ReadVariableOp&model_2/dense_10/MatMul/ReadVariableOp2P
&model_2/dense_11/MatMul/ReadVariableOp&model_2/dense_11/MatMul/ReadVariableOp2P
&model_2/dense_12/MatMul/ReadVariableOp&model_2/dense_12/MatMul/ReadVariableOp2P
&model_2/dense_13/MatMul/ReadVariableOp&model_2/dense_13/MatMul/ReadVariableOp2P
&model_2/dense_14/MatMul/ReadVariableOp&model_2/dense_14/MatMul/ReadVariableOp2P
&model_2/dense_15/MatMul/ReadVariableOp&model_2/dense_15/MatMul/ReadVariableOp2P
&model_2/dense_16/MatMul/ReadVariableOp&model_2/dense_16/MatMul/ReadVariableOp2P
&model_2/dense_17/MatMul/ReadVariableOp&model_2/dense_17/MatMul/ReadVariableOp2P
&model_2/dense_18/MatMul/ReadVariableOp&model_2/dense_18/MatMul/ReadVariableOp2P
&model_2/dense_19/MatMul/ReadVariableOp&model_2/dense_19/MatMul/ReadVariableOp2T
(model_2/net_output/MatMul/ReadVariableOp(model_2/net_output/MatMul/ReadVariableOp:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
с
E
.__inference_dense_17_activity_regularizer_4458
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
яЋ
ћ
A__inference_model_2_layer_call_and_return_conditional_losses_5670
input_2
dense_10_5511:
dense_11_5522:
dense_12_5534:
dense_13_5546:!
net_output_5558:
dense_14_5577:
dense_15_5589:
dense_16_5601:
dense_17_5613:
dense_18_5625:
dense_19_5637:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12ѕб dense_10/StatefulPartitionedCallб dense_11/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб dense_13/StatefulPartitionedCallб dense_14/StatefulPartitionedCallб dense_15/StatefulPartitionedCallб dense_16/StatefulPartitionedCallб dense_17/StatefulPartitionedCallб dense_18/StatefulPartitionedCallб dense_19/StatefulPartitionedCallб"net_output/StatefulPartitionedCall■
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_10_5511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_44992"
 dense_10/StatefulPartitionedCallЭ
,dense_10/ActivityRegularizer/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_10_activity_regularizer_43542.
,dense_10/ActivityRegularizer/PartitionedCallА
"dense_10/ActivityRegularizer/ShapeShape)dense_10/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_10/ActivityRegularizer/Shape«
0dense_10/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_10/ActivityRegularizer/strided_slice/stack▓
2dense_10/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_1▓
2dense_10/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_2љ
*dense_10/ActivityRegularizer/strided_sliceStridedSlice+dense_10/ActivityRegularizer/Shape:output:09dense_10/ActivityRegularizer/strided_slice/stack:output:0;dense_10/ActivityRegularizer/strided_slice/stack_1:output:0;dense_10/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_10/ActivityRegularizer/strided_slice│
!dense_10/ActivityRegularizer/CastCast3dense_10/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_10/ActivityRegularizer/Castо
$dense_10/ActivityRegularizer/truedivRealDiv5dense_10/ActivityRegularizer/PartitionedCall:output:0%dense_10/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_10/ActivityRegularizer/truedivа
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_5522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_45192"
 dense_11/StatefulPartitionedCallЭ
,dense_11/ActivityRegularizer/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_11_activity_regularizer_43672.
,dense_11/ActivityRegularizer/PartitionedCallА
"dense_11/ActivityRegularizer/ShapeShape)dense_11/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_11/ActivityRegularizer/Shape«
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_11/ActivityRegularizer/strided_slice/stack▓
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_1▓
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_2љ
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_11/ActivityRegularizer/strided_slice│
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_11/ActivityRegularizer/Castо
$dense_11/ActivityRegularizer/truedivRealDiv5dense_11/ActivityRegularizer/PartitionedCall:output:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_11/ActivityRegularizer/truedivЭ
dropout_8/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_8_layer_call_and_return_conditional_losses_45362
dropout_8/PartitionedCallЎ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_12_5534*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_45462"
 dense_12/StatefulPartitionedCallЭ
,dense_12/ActivityRegularizer/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_12_activity_regularizer_43802.
,dense_12/ActivityRegularizer/PartitionedCallА
"dense_12/ActivityRegularizer/ShapeShape)dense_12/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_12/ActivityRegularizer/Shape«
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_12/ActivityRegularizer/strided_slice/stack▓
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_1▓
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_2љ
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_12/ActivityRegularizer/strided_slice│
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_12/ActivityRegularizer/Castо
$dense_12/ActivityRegularizer/truedivRealDiv5dense_12/ActivityRegularizer/PartitionedCall:output:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_12/ActivityRegularizer/truedivЭ
dropout_9/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_9_layer_call_and_return_conditional_losses_45632
dropout_9/PartitionedCallЎ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_13_5546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_45732"
 dense_13/StatefulPartitionedCallЭ
,dense_13/ActivityRegularizer/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_13_activity_regularizer_43932.
,dense_13/ActivityRegularizer/PartitionedCallА
"dense_13/ActivityRegularizer/ShapeShape)dense_13/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_13/ActivityRegularizer/Shape«
0dense_13/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_13/ActivityRegularizer/strided_slice/stack▓
2dense_13/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_1▓
2dense_13/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_2љ
*dense_13/ActivityRegularizer/strided_sliceStridedSlice+dense_13/ActivityRegularizer/Shape:output:09dense_13/ActivityRegularizer/strided_slice/stack:output:0;dense_13/ActivityRegularizer/strided_slice/stack_1:output:0;dense_13/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_13/ActivityRegularizer/strided_slice│
!dense_13/ActivityRegularizer/CastCast3dense_13/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_13/ActivityRegularizer/Castо
$dense_13/ActivityRegularizer/truedivRealDiv5dense_13/ActivityRegularizer/PartitionedCall:output:0%dense_13/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_13/ActivityRegularizer/truedivч
dropout_10/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_10_layer_call_and_return_conditional_losses_45902
dropout_10/PartitionedCallб
"net_output/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0net_output_5558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_net_output_layer_call_and_return_conditional_losses_46002$
"net_output/StatefulPartitionedCallђ
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *9
f4R2
0__inference_net_output_activity_regularizer_440620
.net_output/ActivityRegularizer/PartitionedCallД
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2&
$net_output/ActivityRegularizer/Shape▓
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2net_output/ActivityRegularizer/strided_slice/stackХ
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_1Х
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_2ю
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,net_output/ActivityRegularizer/strided_slice╣
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#net_output/ActivityRegularizer/Castя
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&net_output/ActivityRegularizer/truedivЂ
tf.math.subtract_2/Sub/yConst*
_output_shapes
:*
dtype0*
valueB*    2
tf.math.subtract_2/Sub/y╣
tf.math.subtract_2/SubSub+net_output/StatefulPartitionedCall:output:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:         2
tf.math.subtract_2/Subo
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.pow_1/Pow/yЎ
tf.math.pow_1/PowPowtf.math.subtract_2/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*'
_output_shapes
:         2
tf.math.pow_1/PowБ
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indicesх
tf.math.reduce_sum_1/SumSumtf.math.pow_1/Pow:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_1/Sumё
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constф
tf.math.reduce_mean_2/MeanMean!tf.math.reduce_sum_1/Sum:output:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/Meanб
 dense_14/StatefulPartitionedCallStatefulPartitionedCall+net_output/StatefulPartitionedCall:output:0dense_14_5577*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_46282"
 dense_14/StatefulPartitionedCallЭ
,dense_14/ActivityRegularizer/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_14_activity_regularizer_44192.
,dense_14/ActivityRegularizer/PartitionedCallА
"dense_14/ActivityRegularizer/ShapeShape)dense_14/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_14/ActivityRegularizer/Shape«
0dense_14/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_14/ActivityRegularizer/strided_slice/stack▓
2dense_14/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_1▓
2dense_14/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_2љ
*dense_14/ActivityRegularizer/strided_sliceStridedSlice+dense_14/ActivityRegularizer/Shape:output:09dense_14/ActivityRegularizer/strided_slice/stack:output:0;dense_14/ActivityRegularizer/strided_slice/stack_1:output:0;dense_14/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_14/ActivityRegularizer/strided_slice│
!dense_14/ActivityRegularizer/CastCast3dense_14/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_14/ActivityRegularizer/Castо
$dense_14/ActivityRegularizer/truedivRealDiv5dense_14/ActivityRegularizer/PartitionedCall:output:0%dense_14/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_14/ActivityRegularizer/truedivч
dropout_11/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_46452
dropout_11/PartitionedCallџ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_15_5589*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_46552"
 dense_15/StatefulPartitionedCallЭ
,dense_15/ActivityRegularizer/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_15_activity_regularizer_44322.
,dense_15/ActivityRegularizer/PartitionedCallА
"dense_15/ActivityRegularizer/ShapeShape)dense_15/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_15/ActivityRegularizer/Shape«
0dense_15/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_15/ActivityRegularizer/strided_slice/stack▓
2dense_15/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_1▓
2dense_15/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_2љ
*dense_15/ActivityRegularizer/strided_sliceStridedSlice+dense_15/ActivityRegularizer/Shape:output:09dense_15/ActivityRegularizer/strided_slice/stack:output:0;dense_15/ActivityRegularizer/strided_slice/stack_1:output:0;dense_15/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_15/ActivityRegularizer/strided_slice│
!dense_15/ActivityRegularizer/CastCast3dense_15/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_15/ActivityRegularizer/Castо
$dense_15/ActivityRegularizer/truedivRealDiv5dense_15/ActivityRegularizer/PartitionedCall:output:0%dense_15/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_15/ActivityRegularizer/truedivч
dropout_12/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_12_layer_call_and_return_conditional_losses_46722
dropout_12/PartitionedCallџ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_16_5601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_46822"
 dense_16/StatefulPartitionedCallЭ
,dense_16/ActivityRegularizer/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_16_activity_regularizer_44452.
,dense_16/ActivityRegularizer/PartitionedCallА
"dense_16/ActivityRegularizer/ShapeShape)dense_16/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape«
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack▓
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1▓
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2љ
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice│
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Castо
$dense_16/ActivityRegularizer/truedivRealDiv5dense_16/ActivityRegularizer/PartitionedCall:output:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_16/ActivityRegularizer/truedivч
dropout_13/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_13_layer_call_and_return_conditional_losses_46992
dropout_13/PartitionedCallџ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_17_5613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_47092"
 dense_17/StatefulPartitionedCallЭ
,dense_17/ActivityRegularizer/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_17_activity_regularizer_44582.
,dense_17/ActivityRegularizer/PartitionedCallА
"dense_17/ActivityRegularizer/ShapeShape)dense_17/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_17/ActivityRegularizer/Shape«
0dense_17/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_17/ActivityRegularizer/strided_slice/stack▓
2dense_17/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_1▓
2dense_17/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_2љ
*dense_17/ActivityRegularizer/strided_sliceStridedSlice+dense_17/ActivityRegularizer/Shape:output:09dense_17/ActivityRegularizer/strided_slice/stack:output:0;dense_17/ActivityRegularizer/strided_slice/stack_1:output:0;dense_17/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_17/ActivityRegularizer/strided_slice│
!dense_17/ActivityRegularizer/CastCast3dense_17/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_17/ActivityRegularizer/Castо
$dense_17/ActivityRegularizer/truedivRealDiv5dense_17/ActivityRegularizer/PartitionedCall:output:0%dense_17/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_17/ActivityRegularizer/truedivч
dropout_14/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_14_layer_call_and_return_conditional_losses_47262
dropout_14/PartitionedCallџ
 dense_18/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0dense_18_5625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_47362"
 dense_18/StatefulPartitionedCallЭ
,dense_18/ActivityRegularizer/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_18_activity_regularizer_44712.
,dense_18/ActivityRegularizer/PartitionedCallА
"dense_18/ActivityRegularizer/ShapeShape)dense_18/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_18/ActivityRegularizer/Shape«
0dense_18/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_18/ActivityRegularizer/strided_slice/stack▓
2dense_18/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_1▓
2dense_18/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_2љ
*dense_18/ActivityRegularizer/strided_sliceStridedSlice+dense_18/ActivityRegularizer/Shape:output:09dense_18/ActivityRegularizer/strided_slice/stack:output:0;dense_18/ActivityRegularizer/strided_slice/stack_1:output:0;dense_18/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_18/ActivityRegularizer/strided_slice│
!dense_18/ActivityRegularizer/CastCast3dense_18/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_18/ActivityRegularizer/Castо
$dense_18/ActivityRegularizer/truedivRealDiv5dense_18/ActivityRegularizer/PartitionedCall:output:0%dense_18/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_18/ActivityRegularizer/truedivч
dropout_15/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_15_layer_call_and_return_conditional_losses_47532
dropout_15/PartitionedCallџ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_19_5637*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_47632"
 dense_19/StatefulPartitionedCallЭ
,dense_19/ActivityRegularizer/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_19_activity_regularizer_44842.
,dense_19/ActivityRegularizer/PartitionedCallА
"dense_19/ActivityRegularizer/ShapeShape)dense_19/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_19/ActivityRegularizer/Shape«
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_19/ActivityRegularizer/strided_slice/stack▓
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_1▓
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_2љ
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_19/ActivityRegularizer/strided_slice│
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_19/ActivityRegularizer/Castо
$dense_19/ActivityRegularizer/truedivRealDiv5dense_19/ActivityRegularizer/PartitionedCall:output:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_19/ActivityRegularizer/truedivЮ
tf.math.subtract_3/SubSub)dense_19/StatefulPartitionedCall:output:0input_2*
T0*'
_output_shapes
:         2
tf.math.subtract_3/Subі
tf.math.square_1/SquareSquaretf.math.subtract_3/Sub:z:0*
T0*'
_output_shapes
:         2
tf.math.square_1/SquareІ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constц
tf.math.reduce_mean_3/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean░
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_2/Mean:output:0#tf.math.reduce_mean_3/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_2/AddV2y
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 */Lь62
tf.__operators__.add_3/yФ
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_3/AddV2С
add_loss_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_add_loss_1_layer_call_and_return_conditional_losses_47872
add_loss_1/PartitionedCallx
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityv

Identity_1Identity(dense_10/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1v

Identity_2Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_2v

Identity_3Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3v

Identity_4Identity(dense_13/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_4x

Identity_5Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_5v

Identity_6Identity(dense_14/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_6v

Identity_7Identity(dense_15/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_7v

Identity_8Identity(dense_16/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_8v

Identity_9Identity(dense_17/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_9x
Identity_10Identity(dense_18/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_10x
Identity_11Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_11s
Identity_12Identity#add_loss_1/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 2
Identity_12Л
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
─
{
'__inference_dense_19_layer_call_fn_6823

inputs
unknown:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_47632
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е
b
C__inference_dropout_9_layer_call_and_return_conditional_losses_5149

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_11_layer_call_and_return_conditional_losses_6613

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_14_layer_call_and_return_conditional_losses_6754

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_11_layer_call_and_return_conditional_losses_5049

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ч
E
)__inference_add_loss_1_layer_call_fn_6834

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_add_loss_1_layer_call_and_return_conditional_losses_47872
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
Ђч
Џ

A__inference_model_2_layer_call_and_return_conditional_losses_6353

inputs9
'dense_10_matmul_readvariableop_resource:9
'dense_11_matmul_readvariableop_resource:9
'dense_12_matmul_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:;
)net_output_matmul_readvariableop_resource:9
'dense_14_matmul_readvariableop_resource:9
'dense_15_matmul_readvariableop_resource:9
'dense_16_matmul_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:9
'dense_18_matmul_readvariableop_resource:9
'dense_19_matmul_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12ѕбdense_10/MatMul/ReadVariableOpбdense_11/MatMul/ReadVariableOpбdense_12/MatMul/ReadVariableOpбdense_13/MatMul/ReadVariableOpбdense_14/MatMul/ReadVariableOpбdense_15/MatMul/ReadVariableOpбdense_16/MatMul/ReadVariableOpбdense_17/MatMul/ReadVariableOpбdense_18/MatMul/ReadVariableOpбdense_19/MatMul/ReadVariableOpб net_output/MatMul/ReadVariableOpе
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_10/MatMul/ReadVariableOpј
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_10/MatMuls
dense_10/ReluReludense_10/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_10/ReluБ
#dense_10/ActivityRegularizer/SquareSquaredense_10/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_10/ActivityRegularizer/SquareЎ
"dense_10/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_10/ActivityRegularizer/Const┬
 dense_10/ActivityRegularizer/SumSum'dense_10/ActivityRegularizer/Square:y:0+dense_10/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_10/ActivityRegularizer/SumЇ
"dense_10/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_10/ActivityRegularizer/mul/x─
 dense_10/ActivityRegularizer/mulMul+dense_10/ActivityRegularizer/mul/x:output:0)dense_10/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_10/ActivityRegularizer/mulЊ
"dense_10/ActivityRegularizer/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_10/ActivityRegularizer/Shape«
0dense_10/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_10/ActivityRegularizer/strided_slice/stack▓
2dense_10/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_1▓
2dense_10/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_10/ActivityRegularizer/strided_slice/stack_2љ
*dense_10/ActivityRegularizer/strided_sliceStridedSlice+dense_10/ActivityRegularizer/Shape:output:09dense_10/ActivityRegularizer/strided_slice/stack:output:0;dense_10/ActivityRegularizer/strided_slice/stack_1:output:0;dense_10/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_10/ActivityRegularizer/strided_slice│
!dense_10/ActivityRegularizer/CastCast3dense_10/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_10/ActivityRegularizer/Cast┼
$dense_10/ActivityRegularizer/truedivRealDiv$dense_10/ActivityRegularizer/mul:z:0%dense_10/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_10/ActivityRegularizer/truedivе
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOpБ
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/MatMuls
dense_11/ReluReludense_11/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_11/ReluБ
#dense_11/ActivityRegularizer/SquareSquaredense_11/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_11/ActivityRegularizer/SquareЎ
"dense_11/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_11/ActivityRegularizer/Const┬
 dense_11/ActivityRegularizer/SumSum'dense_11/ActivityRegularizer/Square:y:0+dense_11/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_11/ActivityRegularizer/SumЇ
"dense_11/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_11/ActivityRegularizer/mul/x─
 dense_11/ActivityRegularizer/mulMul+dense_11/ActivityRegularizer/mul/x:output:0)dense_11/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_11/ActivityRegularizer/mulЊ
"dense_11/ActivityRegularizer/ShapeShapedense_11/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_11/ActivityRegularizer/Shape«
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_11/ActivityRegularizer/strided_slice/stack▓
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_1▓
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_11/ActivityRegularizer/strided_slice/stack_2љ
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_11/ActivityRegularizer/strided_slice│
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_11/ActivityRegularizer/Cast┼
$dense_11/ActivityRegularizer/truedivRealDiv$dense_11/ActivityRegularizer/mul:z:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_11/ActivityRegularizer/truedivw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_8/dropout/Constд
dropout_8/dropout/MulMuldense_11/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_8/dropout/Mul}
dropout_8/dropout/ShapeShapedense_11/Relu:activations:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shapeм
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype020
.dropout_8/dropout/random_uniform/RandomUniformЅ
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2"
 dropout_8/dropout/GreaterEqual/yТ
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2 
dropout_8/dropout/GreaterEqualЮ
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_8/dropout/Castб
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_8/dropout/Mul_1е
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOpБ
dense_12/MatMulMatMuldropout_8/dropout/Mul_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_12/MatMuls
dense_12/ReluReludense_12/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_12/ReluБ
#dense_12/ActivityRegularizer/SquareSquaredense_12/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_12/ActivityRegularizer/SquareЎ
"dense_12/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_12/ActivityRegularizer/Const┬
 dense_12/ActivityRegularizer/SumSum'dense_12/ActivityRegularizer/Square:y:0+dense_12/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_12/ActivityRegularizer/SumЇ
"dense_12/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_12/ActivityRegularizer/mul/x─
 dense_12/ActivityRegularizer/mulMul+dense_12/ActivityRegularizer/mul/x:output:0)dense_12/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_12/ActivityRegularizer/mulЊ
"dense_12/ActivityRegularizer/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_12/ActivityRegularizer/Shape«
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_12/ActivityRegularizer/strided_slice/stack▓
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_1▓
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_12/ActivityRegularizer/strided_slice/stack_2љ
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_12/ActivityRegularizer/strided_slice│
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_12/ActivityRegularizer/Cast┼
$dense_12/ActivityRegularizer/truedivRealDiv$dense_12/ActivityRegularizer/mul:z:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_12/ActivityRegularizer/truedivw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_9/dropout/Constд
dropout_9/dropout/MulMuldense_12/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_9/dropout/Mul}
dropout_9/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shapeм
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype020
.dropout_9/dropout/random_uniform/RandomUniformЅ
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2"
 dropout_9/dropout/GreaterEqual/yТ
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2 
dropout_9/dropout/GreaterEqualЮ
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_9/dropout/Castб
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_9/dropout/Mul_1е
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_13/MatMul/ReadVariableOpБ
dense_13/MatMulMatMuldropout_9/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/MatMuls
dense_13/ReluReludense_13/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_13/ReluБ
#dense_13/ActivityRegularizer/SquareSquaredense_13/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_13/ActivityRegularizer/SquareЎ
"dense_13/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_13/ActivityRegularizer/Const┬
 dense_13/ActivityRegularizer/SumSum'dense_13/ActivityRegularizer/Square:y:0+dense_13/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_13/ActivityRegularizer/SumЇ
"dense_13/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_13/ActivityRegularizer/mul/x─
 dense_13/ActivityRegularizer/mulMul+dense_13/ActivityRegularizer/mul/x:output:0)dense_13/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_13/ActivityRegularizer/mulЊ
"dense_13/ActivityRegularizer/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_13/ActivityRegularizer/Shape«
0dense_13/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_13/ActivityRegularizer/strided_slice/stack▓
2dense_13/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_1▓
2dense_13/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_13/ActivityRegularizer/strided_slice/stack_2љ
*dense_13/ActivityRegularizer/strided_sliceStridedSlice+dense_13/ActivityRegularizer/Shape:output:09dense_13/ActivityRegularizer/strided_slice/stack:output:0;dense_13/ActivityRegularizer/strided_slice/stack_1:output:0;dense_13/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_13/ActivityRegularizer/strided_slice│
!dense_13/ActivityRegularizer/CastCast3dense_13/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_13/ActivityRegularizer/Cast┼
$dense_13/ActivityRegularizer/truedivRealDiv$dense_13/ActivityRegularizer/mul:z:0%dense_13/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_13/ActivityRegularizer/truedivy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_10/dropout/ConstЕ
dropout_10/dropout/MulMuldense_13/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_10/dropout/Mul
dropout_10/dropout/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:2
dropout_10/dropout/ShapeН
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype021
/dropout_10/dropout/random_uniform/RandomUniformІ
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_10/dropout/GreaterEqual/yЖ
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2!
dropout_10/dropout/GreaterEqualа
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_10/dropout/Castд
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_10/dropout/Mul_1«
 net_output/MatMul/ReadVariableOpReadVariableOp)net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 net_output/MatMul/ReadVariableOpф
net_output/MatMulMatMuldropout_10/dropout/Mul_1:z:0(net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
net_output/MatMuly
net_output/ReluRelunet_output/MatMul:product:0*
T0*'
_output_shapes
:         2
net_output/ReluЕ
%net_output/ActivityRegularizer/SquareSquarenet_output/Relu:activations:0*
T0*'
_output_shapes
:         2'
%net_output/ActivityRegularizer/SquareЮ
$net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$net_output/ActivityRegularizer/Const╩
"net_output/ActivityRegularizer/SumSum)net_output/ActivityRegularizer/Square:y:0-net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2$
"net_output/ActivityRegularizer/SumЉ
$net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2&
$net_output/ActivityRegularizer/mul/x╠
"net_output/ActivityRegularizer/mulMul-net_output/ActivityRegularizer/mul/x:output:0+net_output/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"net_output/ActivityRegularizer/mulЎ
$net_output/ActivityRegularizer/ShapeShapenet_output/Relu:activations:0*
T0*
_output_shapes
:2&
$net_output/ActivityRegularizer/Shape▓
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2net_output/ActivityRegularizer/strided_slice/stackХ
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_1Х
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4net_output/ActivityRegularizer/strided_slice/stack_2ю
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,net_output/ActivityRegularizer/strided_slice╣
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#net_output/ActivityRegularizer/Cast═
&net_output/ActivityRegularizer/truedivRealDiv&net_output/ActivityRegularizer/mul:z:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&net_output/ActivityRegularizer/truedivЂ
tf.math.subtract_2/Sub/yConst*
_output_shapes
:*
dtype0*
valueB*    2
tf.math.subtract_2/Sub/yФ
tf.math.subtract_2/SubSubnet_output/Relu:activations:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:         2
tf.math.subtract_2/Subo
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.pow_1/Pow/yЎ
tf.math.pow_1/PowPowtf.math.subtract_2/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*'
_output_shapes
:         2
tf.math.pow_1/PowБ
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*tf.math.reduce_sum_1/Sum/reduction_indicesх
tf.math.reduce_sum_1/SumSumtf.math.pow_1/Pow:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
tf.math.reduce_sum_1/Sumё
tf.math.reduce_mean_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_2/Constф
tf.math.reduce_mean_2/MeanMean!tf.math.reduce_sum_1/Sum:output:0$tf.math.reduce_mean_2/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_2/Meanе
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_14/MatMul/ReadVariableOpЦ
dense_14/MatMulMatMulnet_output/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMuls
dense_14/ReluReludense_14/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_14/ReluБ
#dense_14/ActivityRegularizer/SquareSquaredense_14/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_14/ActivityRegularizer/SquareЎ
"dense_14/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_14/ActivityRegularizer/Const┬
 dense_14/ActivityRegularizer/SumSum'dense_14/ActivityRegularizer/Square:y:0+dense_14/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_14/ActivityRegularizer/SumЇ
"dense_14/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_14/ActivityRegularizer/mul/x─
 dense_14/ActivityRegularizer/mulMul+dense_14/ActivityRegularizer/mul/x:output:0)dense_14/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_14/ActivityRegularizer/mulЊ
"dense_14/ActivityRegularizer/ShapeShapedense_14/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_14/ActivityRegularizer/Shape«
0dense_14/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_14/ActivityRegularizer/strided_slice/stack▓
2dense_14/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_1▓
2dense_14/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_14/ActivityRegularizer/strided_slice/stack_2љ
*dense_14/ActivityRegularizer/strided_sliceStridedSlice+dense_14/ActivityRegularizer/Shape:output:09dense_14/ActivityRegularizer/strided_slice/stack:output:0;dense_14/ActivityRegularizer/strided_slice/stack_1:output:0;dense_14/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_14/ActivityRegularizer/strided_slice│
!dense_14/ActivityRegularizer/CastCast3dense_14/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_14/ActivityRegularizer/Cast┼
$dense_14/ActivityRegularizer/truedivRealDiv$dense_14/ActivityRegularizer/mul:z:0%dense_14/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_14/ActivityRegularizer/truedivy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_11/dropout/ConstЕ
dropout_11/dropout/MulMuldense_14/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapedense_14/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/ShapeН
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype021
/dropout_11/dropout/random_uniform/RandomUniformІ
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_11/dropout/GreaterEqual/yЖ
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2!
dropout_11/dropout/GreaterEqualа
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_11/dropout/Castд
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_11/dropout/Mul_1е
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_15/MatMul/ReadVariableOpц
dense_15/MatMulMatMuldropout_11/dropout/Mul_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/MatMuls
dense_15/ReluReludense_15/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_15/ReluБ
#dense_15/ActivityRegularizer/SquareSquaredense_15/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_15/ActivityRegularizer/SquareЎ
"dense_15/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_15/ActivityRegularizer/Const┬
 dense_15/ActivityRegularizer/SumSum'dense_15/ActivityRegularizer/Square:y:0+dense_15/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_15/ActivityRegularizer/SumЇ
"dense_15/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_15/ActivityRegularizer/mul/x─
 dense_15/ActivityRegularizer/mulMul+dense_15/ActivityRegularizer/mul/x:output:0)dense_15/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_15/ActivityRegularizer/mulЊ
"dense_15/ActivityRegularizer/ShapeShapedense_15/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_15/ActivityRegularizer/Shape«
0dense_15/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_15/ActivityRegularizer/strided_slice/stack▓
2dense_15/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_1▓
2dense_15/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_15/ActivityRegularizer/strided_slice/stack_2љ
*dense_15/ActivityRegularizer/strided_sliceStridedSlice+dense_15/ActivityRegularizer/Shape:output:09dense_15/ActivityRegularizer/strided_slice/stack:output:0;dense_15/ActivityRegularizer/strided_slice/stack_1:output:0;dense_15/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_15/ActivityRegularizer/strided_slice│
!dense_15/ActivityRegularizer/CastCast3dense_15/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_15/ActivityRegularizer/Cast┼
$dense_15/ActivityRegularizer/truedivRealDiv$dense_15/ActivityRegularizer/mul:z:0%dense_15/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_15/ActivityRegularizer/truedivy
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_12/dropout/ConstЕ
dropout_12/dropout/MulMuldense_15/Relu:activations:0!dropout_12/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_12/dropout/Mul
dropout_12/dropout/ShapeShapedense_15/Relu:activations:0*
T0*
_output_shapes
:2
dropout_12/dropout/ShapeН
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype021
/dropout_12/dropout/random_uniform/RandomUniformІ
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_12/dropout/GreaterEqual/yЖ
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2!
dropout_12/dropout/GreaterEqualа
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_12/dropout/Castд
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_12/dropout/Mul_1е
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_16/MatMul/ReadVariableOpц
dense_16/MatMulMatMuldropout_12/dropout/Mul_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_16/MatMuls
dense_16/ReluReludense_16/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_16/ReluБ
#dense_16/ActivityRegularizer/SquareSquaredense_16/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_16/ActivityRegularizer/SquareЎ
"dense_16/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_16/ActivityRegularizer/Const┬
 dense_16/ActivityRegularizer/SumSum'dense_16/ActivityRegularizer/Square:y:0+dense_16/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_16/ActivityRegularizer/SumЇ
"dense_16/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_16/ActivityRegularizer/mul/x─
 dense_16/ActivityRegularizer/mulMul+dense_16/ActivityRegularizer/mul/x:output:0)dense_16/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_16/ActivityRegularizer/mulЊ
"dense_16/ActivityRegularizer/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape«
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack▓
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1▓
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2љ
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice│
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Cast┼
$dense_16/ActivityRegularizer/truedivRealDiv$dense_16/ActivityRegularizer/mul:z:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_16/ActivityRegularizer/truedivy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_13/dropout/ConstЕ
dropout_13/dropout/MulMuldense_16/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_13/dropout/Mul
dropout_13/dropout/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/ShapeН
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype021
/dropout_13/dropout/random_uniform/RandomUniformІ
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_13/dropout/GreaterEqual/yЖ
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2!
dropout_13/dropout/GreaterEqualа
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_13/dropout/Castд
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_13/dropout/Mul_1е
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOpц
dense_17/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_17/MatMuls
dense_17/ReluReludense_17/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_17/ReluБ
#dense_17/ActivityRegularizer/SquareSquaredense_17/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_17/ActivityRegularizer/SquareЎ
"dense_17/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_17/ActivityRegularizer/Const┬
 dense_17/ActivityRegularizer/SumSum'dense_17/ActivityRegularizer/Square:y:0+dense_17/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_17/ActivityRegularizer/SumЇ
"dense_17/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_17/ActivityRegularizer/mul/x─
 dense_17/ActivityRegularizer/mulMul+dense_17/ActivityRegularizer/mul/x:output:0)dense_17/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_17/ActivityRegularizer/mulЊ
"dense_17/ActivityRegularizer/ShapeShapedense_17/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_17/ActivityRegularizer/Shape«
0dense_17/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_17/ActivityRegularizer/strided_slice/stack▓
2dense_17/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_1▓
2dense_17/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_17/ActivityRegularizer/strided_slice/stack_2љ
*dense_17/ActivityRegularizer/strided_sliceStridedSlice+dense_17/ActivityRegularizer/Shape:output:09dense_17/ActivityRegularizer/strided_slice/stack:output:0;dense_17/ActivityRegularizer/strided_slice/stack_1:output:0;dense_17/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_17/ActivityRegularizer/strided_slice│
!dense_17/ActivityRegularizer/CastCast3dense_17/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_17/ActivityRegularizer/Cast┼
$dense_17/ActivityRegularizer/truedivRealDiv$dense_17/ActivityRegularizer/mul:z:0%dense_17/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_17/ActivityRegularizer/truedivy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_14/dropout/ConstЕ
dropout_14/dropout/MulMuldense_17/Relu:activations:0!dropout_14/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_14/dropout/Mul
dropout_14/dropout/ShapeShapedense_17/Relu:activations:0*
T0*
_output_shapes
:2
dropout_14/dropout/ShapeН
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype021
/dropout_14/dropout/random_uniform/RandomUniformІ
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_14/dropout/GreaterEqual/yЖ
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2!
dropout_14/dropout/GreaterEqualа
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_14/dropout/Castд
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_14/dropout/Mul_1е
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_18/MatMul/ReadVariableOpц
dense_18/MatMulMatMuldropout_14/dropout/Mul_1:z:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_18/MatMuls
dense_18/ReluReludense_18/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_18/ReluБ
#dense_18/ActivityRegularizer/SquareSquaredense_18/Relu:activations:0*
T0*'
_output_shapes
:         2%
#dense_18/ActivityRegularizer/SquareЎ
"dense_18/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_18/ActivityRegularizer/Const┬
 dense_18/ActivityRegularizer/SumSum'dense_18/ActivityRegularizer/Square:y:0+dense_18/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_18/ActivityRegularizer/SumЇ
"dense_18/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_18/ActivityRegularizer/mul/x─
 dense_18/ActivityRegularizer/mulMul+dense_18/ActivityRegularizer/mul/x:output:0)dense_18/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_18/ActivityRegularizer/mulЊ
"dense_18/ActivityRegularizer/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2$
"dense_18/ActivityRegularizer/Shape«
0dense_18/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_18/ActivityRegularizer/strided_slice/stack▓
2dense_18/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_1▓
2dense_18/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_18/ActivityRegularizer/strided_slice/stack_2љ
*dense_18/ActivityRegularizer/strided_sliceStridedSlice+dense_18/ActivityRegularizer/Shape:output:09dense_18/ActivityRegularizer/strided_slice/stack:output:0;dense_18/ActivityRegularizer/strided_slice/stack_1:output:0;dense_18/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_18/ActivityRegularizer/strided_slice│
!dense_18/ActivityRegularizer/CastCast3dense_18/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_18/ActivityRegularizer/Cast┼
$dense_18/ActivityRegularizer/truedivRealDiv$dense_18/ActivityRegularizer/mul:z:0%dense_18/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_18/ActivityRegularizer/truedivy
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_15/dropout/ConstЕ
dropout_15/dropout/MulMuldense_18/Relu:activations:0!dropout_15/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_15/dropout/Mul
dropout_15/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2
dropout_15/dropout/ShapeН
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype021
/dropout_15/dropout/random_uniform/RandomUniformІ
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_15/dropout/GreaterEqual/yЖ
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2!
dropout_15/dropout/GreaterEqualа
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_15/dropout/Castд
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_15/dropout/Mul_1е
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_19/MatMul/ReadVariableOpц
dense_19/MatMulMatMuldropout_15/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_19/MatMul|
dense_19/SigmoidSigmoiddense_19/MatMul:product:0*
T0*'
_output_shapes
:         2
dense_19/Sigmoidю
#dense_19/ActivityRegularizer/SquareSquaredense_19/Sigmoid:y:0*
T0*'
_output_shapes
:         2%
#dense_19/ActivityRegularizer/SquareЎ
"dense_19/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_19/ActivityRegularizer/Const┬
 dense_19/ActivityRegularizer/SumSum'dense_19/ActivityRegularizer/Square:y:0+dense_19/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_19/ActivityRegularizer/SumЇ
"dense_19/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2$
"dense_19/ActivityRegularizer/mul/x─
 dense_19/ActivityRegularizer/mulMul+dense_19/ActivityRegularizer/mul/x:output:0)dense_19/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_19/ActivityRegularizer/mulї
"dense_19/ActivityRegularizer/ShapeShapedense_19/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_19/ActivityRegularizer/Shape«
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_19/ActivityRegularizer/strided_slice/stack▓
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_1▓
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_19/ActivityRegularizer/strided_slice/stack_2љ
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_19/ActivityRegularizer/strided_slice│
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_19/ActivityRegularizer/Cast┼
$dense_19/ActivityRegularizer/truedivRealDiv$dense_19/ActivityRegularizer/mul:z:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_19/ActivityRegularizer/truedivЄ
tf.math.subtract_3/SubSubdense_19/Sigmoid:y:0inputs*
T0*'
_output_shapes
:         2
tf.math.subtract_3/Subі
tf.math.square_1/SquareSquaretf.math.subtract_3/Sub:z:0*
T0*'
_output_shapes
:         2
tf.math.square_1/SquareІ
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean_3/Constц
tf.math.reduce_mean_3/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean░
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_2/Mean:output:0#tf.math.reduce_mean_3/Mean:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_2/AddV2y
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 */Lь62
tf.__operators__.add_3/yФ
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: 2
tf.__operators__.add_3/AddV2x
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityv

Identity_1Identity(dense_10/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1v

Identity_2Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_2v

Identity_3Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3v

Identity_4Identity(dense_13/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_4x

Identity_5Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_5v

Identity_6Identity(dense_14/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_6v

Identity_7Identity(dense_15/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_7v

Identity_8Identity(dense_16/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_8v

Identity_9Identity(dense_17/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_9x
Identity_10Identity(dense_18/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_10x
Identity_11Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_11p
Identity_12Identity tf.__operators__.add_3/AddV2:z:0^NoOp*
T0*
_output_shapes
: 2
Identity_12╗
NoOpNoOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_12/MatMul/ReadVariableOp^dense_13/MatMul/ReadVariableOp^dense_14/MatMul/ReadVariableOp^dense_15/MatMul/ReadVariableOp^dense_16/MatMul/ReadVariableOp^dense_17/MatMul/ReadVariableOp^dense_18/MatMul/ReadVariableOp^dense_19/MatMul/ReadVariableOp!^net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2D
 net_output/MatMul/ReadVariableOp net_output/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╝
E
)__inference_dropout_12_layer_call_fn_6673

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_12_layer_call_and_return_conditional_losses_46722
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┌
Ф
B__inference_dense_19_layer_call_and_return_conditional_losses_6922

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMula
SigmoidSigmoidMatMul:product:0*
T0*'
_output_shapes
:         2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е
b
C__inference_dropout_8_layer_call_and_return_conditional_losses_6480

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─
{
'__inference_dense_17_layer_call_fn_6737

inputs
unknown:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_47092
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_12_layer_call_and_return_conditional_losses_6858

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_12_layer_call_and_return_conditional_losses_4672

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_15_layer_call_and_return_conditional_losses_6785

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
­
a
C__inference_dropout_9_layer_call_and_return_conditional_losses_4563

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_15_layer_call_and_return_conditional_losses_4655

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_13_layer_call_and_return_conditional_losses_6699

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_11_layer_call_and_return_conditional_losses_6625

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ъ
b
)__inference_dropout_15_layer_call_fn_6807

inputs
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_15_layer_call_and_return_conditional_losses_48852
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ќ
ю
&__inference_model_2_layer_call_fn_6392

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identityѕбStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         : : : : : : : : : : : : *-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_48032
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ю
a
(__inference_dropout_8_layer_call_fn_6490

inputs
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_8_layer_call_and_return_conditional_losses_51902
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─
{
'__inference_dense_18_layer_call_fn_6780

inputs
unknown:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_47362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─
{
'__inference_dense_11_layer_call_fn_6463

inputs
unknown:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_45192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╝
E
)__inference_dropout_15_layer_call_fn_6802

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_15_layer_call_and_return_conditional_losses_47532
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
­
a
C__inference_dropout_8_layer_call_and_return_conditional_losses_6468

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_18_layer_call_and_return_conditional_losses_6914

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
с
E
.__inference_dense_13_activity_regularizer_4393
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
Ќ
ю
&__inference_model_2_layer_call_fn_6431

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identityѕбStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         : : : : : : : : : : : : *-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_54322
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е
b
C__inference_dropout_8_layer_call_and_return_conditional_losses_5190

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╝
E
)__inference_dropout_14_layer_call_fn_6759

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_14_layer_call_and_return_conditional_losses_47262
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_10_layer_call_and_return_conditional_losses_6554

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
ф
F__inference_dense_18_layer_call_and_return_all_conditional_losses_6773

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_47362
StatefulPartitionedCallх
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_18_activity_regularizer_44712
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╝
E
)__inference_dropout_11_layer_call_fn_6630

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_46452
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ю
a
(__inference_dropout_9_layer_call_fn_6533

inputs
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dropout_9_layer_call_and_return_conditional_losses_51492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
с
E
.__inference_dense_10_activity_regularizer_4354
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
џ
Ю
&__inference_model_2_layer_call_fn_4840
input_2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identityѕбStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         : : : : : : : : : : : : *-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_48032
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
─
{
'__inference_dense_15_layer_call_fn_6651

inputs
unknown:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_46552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
ф
F__inference_dense_16_layer_call_and_return_all_conditional_losses_6687

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_46822
StatefulPartitionedCallх
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_16_activity_regularizer_44452
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ъ
b
)__inference_dropout_14_layer_call_fn_6764

inputs
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_14_layer_call_and_return_conditional_losses_49262
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е
b
C__inference_dropout_9_layer_call_and_return_conditional_losses_6523

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ъ
b
)__inference_dropout_11_layer_call_fn_6635

inputs
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_11_layer_call_and_return_conditional_losses_50492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_13_layer_call_and_return_conditional_losses_4573

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
Ф
B__inference_dense_13_layer_call_and_return_conditional_losses_6866

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_10_layer_call_and_return_conditional_losses_6566

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
c
D__inference_dropout_15_layer_call_and_return_conditional_losses_6797

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
ф
F__inference_dense_11_layer_call_and_return_all_conditional_losses_6456

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_45192
StatefulPartitionedCallх
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *7
f2R0
.__inference_dense_11_activity_regularizer_43672
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
b
D__inference_dropout_14_layer_call_and_return_conditional_losses_4726

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ю«
ч
 __inference__traced_restore_7195
file_prefix2
 assignvariableop_dense_10_kernel:4
"assignvariableop_1_dense_11_kernel:4
"assignvariableop_2_dense_12_kernel:4
"assignvariableop_3_dense_13_kernel:6
$assignvariableop_4_net_output_kernel:4
"assignvariableop_5_dense_14_kernel:4
"assignvariableop_6_dense_15_kernel:4
"assignvariableop_7_dense_16_kernel:4
"assignvariableop_8_dense_17_kernel:4
"assignvariableop_9_dense_18_kernel:5
#assignvariableop_10_dense_19_kernel:'
assignvariableop_11_adam_iter:	 )
assignvariableop_12_adam_beta_1: )
assignvariableop_13_adam_beta_2: (
assignvariableop_14_adam_decay: 0
&assignvariableop_15_adam_learning_rate: #
assignvariableop_16_total: #
assignvariableop_17_count: <
*assignvariableop_18_adam_dense_10_kernel_m:<
*assignvariableop_19_adam_dense_11_kernel_m:<
*assignvariableop_20_adam_dense_12_kernel_m:<
*assignvariableop_21_adam_dense_13_kernel_m:>
,assignvariableop_22_adam_net_output_kernel_m:<
*assignvariableop_23_adam_dense_14_kernel_m:<
*assignvariableop_24_adam_dense_15_kernel_m:<
*assignvariableop_25_adam_dense_16_kernel_m:<
*assignvariableop_26_adam_dense_17_kernel_m:<
*assignvariableop_27_adam_dense_18_kernel_m:<
*assignvariableop_28_adam_dense_19_kernel_m:<
*assignvariableop_29_adam_dense_10_kernel_v:<
*assignvariableop_30_adam_dense_11_kernel_v:<
*assignvariableop_31_adam_dense_12_kernel_v:<
*assignvariableop_32_adam_dense_13_kernel_v:>
,assignvariableop_33_adam_net_output_kernel_v:<
*assignvariableop_34_adam_dense_14_kernel_v:<
*assignvariableop_35_adam_dense_15_kernel_v:<
*assignvariableop_36_adam_dense_16_kernel_v:<
*assignvariableop_37_adam_dense_17_kernel_v:<
*assignvariableop_38_adam_dense_18_kernel_v:<
*assignvariableop_39_adam_dense_19_kernel_v:
identity_41ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Џ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*Д
valueЮBџ)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЯ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesч
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*║
_output_shapesД
ц:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЪ
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Д
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_11_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Д
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_12_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Д
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_13_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Е
AssignVariableOp_4AssignVariableOp$assignvariableop_4_net_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Д
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_14_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Д
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_15_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Д
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_16_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Д
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_17_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Д
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_18_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ф
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_19_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11Ц
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Д
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Д
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14д
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15«
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16А
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17А
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18▓
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_10_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19▓
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_11_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20▓
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_12_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21▓
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_13_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22┤
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_net_output_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23▓
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_14_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24▓
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_15_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▓
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_16_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26▓
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_17_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27▓
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_18_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28▓
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_19_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29▓
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_10_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▓
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_11_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31▓
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_12_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▓
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_13_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33┤
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_net_output_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▓
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_dense_14_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35▓
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_15_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36▓
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_dense_16_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37▓
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_17_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▓
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_18_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_19_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_399
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╬
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_40f
Identity_41IdentityIdentity_40:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_41Х
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_41Identity_41:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЪ
;
input_20
serving_default_input_2:0         D
tf.math.reduce_sum_1,
StatefulPartitionedCall:0         tensorflow/serving/predict:ПЅ
Е
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer_with_weights-9
layer-20
layer-21
layer_with_weights-10
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
	optimizer
 loss
!	variables
"regularization_losses
#trainable_variables
$	keras_api
%
signatures
+Њ&call_and_return_all_conditional_losses
ћ_default_save_signature
Ћ__call__"
_tf_keras_network
"
_tf_keras_input_layer
│

&kernel
'regularization_losses
(	variables
)trainable_variables
*	keras_api
+ќ&call_and_return_all_conditional_losses
Ќ__call__"
_tf_keras_layer
│

+kernel
,regularization_losses
-	variables
.trainable_variables
/	keras_api
+ў&call_and_return_all_conditional_losses
Ў__call__"
_tf_keras_layer
Д
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+џ&call_and_return_all_conditional_losses
Џ__call__"
_tf_keras_layer
│

4kernel
5regularization_losses
6	variables
7trainable_variables
8	keras_api
+ю&call_and_return_all_conditional_losses
Ю__call__"
_tf_keras_layer
Д
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+ъ&call_and_return_all_conditional_losses
Ъ__call__"
_tf_keras_layer
│

=kernel
>regularization_losses
?	variables
@trainable_variables
A	keras_api
+а&call_and_return_all_conditional_losses
А__call__"
_tf_keras_layer
Д
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+б&call_and_return_all_conditional_losses
Б__call__"
_tf_keras_layer
│

Fkernel
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
+ц&call_and_return_all_conditional_losses
Ц__call__"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
(
L	keras_api"
_tf_keras_layer
(
M	keras_api"
_tf_keras_layer
│

Nkernel
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
+д&call_and_return_all_conditional_losses
Д__call__"
_tf_keras_layer
Д
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
+е&call_and_return_all_conditional_losses
Е__call__"
_tf_keras_layer
│

Wkernel
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+ф&call_and_return_all_conditional_losses
Ф__call__"
_tf_keras_layer
Д
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+г&call_and_return_all_conditional_losses
Г__call__"
_tf_keras_layer
│

`kernel
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
+«&call_and_return_all_conditional_losses
»__call__"
_tf_keras_layer
Д
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
+░&call_and_return_all_conditional_losses
▒__call__"
_tf_keras_layer
│

ikernel
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
+▓&call_and_return_all_conditional_losses
│__call__"
_tf_keras_layer
Д
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+┤&call_and_return_all_conditional_losses
х__call__"
_tf_keras_layer
│

rkernel
sregularization_losses
t	variables
utrainable_variables
v	keras_api
+Х&call_and_return_all_conditional_losses
и__call__"
_tf_keras_layer
Д
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
+И&call_and_return_all_conditional_losses
╣__call__"
_tf_keras_layer
│

{kernel
|regularization_losses
}	variables
~trainable_variables
	keras_api
+║&call_and_return_all_conditional_losses
╗__call__"
_tf_keras_layer
)
ђ	keras_api"
_tf_keras_layer
)
Ђ	keras_api"
_tf_keras_layer
)
ѓ	keras_api"
_tf_keras_layer
)
Ѓ	keras_api"
_tf_keras_layer
)
ё	keras_api"
_tf_keras_layer
)
Ё	keras_api"
_tf_keras_layer
Ф
єregularization_losses
Є	variables
ѕtrainable_variables
Ѕ	keras_api
+╝&call_and_return_all_conditional_losses
й__call__"
_tf_keras_layer
┤
	іiter
Іbeta_1
їbeta_2

Їdecay
јlearning_rate&m§+m■4m =mђFmЂNmѓWmЃ`mёimЁrmє{mЄ&vѕ+vЅ4vі=vІFvїNvЇWvј`vЈivљrvЉ{vњ"
	optimizer
 "
trackable_dict_wrapper
n
&0
+1
42
=3
F4
N5
W6
`7
i8
r9
{10"
trackable_list_wrapper
 "
trackable_list_wrapper
n
&0
+1
42
=3
F4
N5
W6
`7
i8
r9
{10"
trackable_list_wrapper
М
Јlayer_metrics
љlayers
!	variables
"regularization_losses
 Љlayer_regularization_losses
#trainable_variables
њnon_trainable_variables
Њmetrics
Ћ__call__
ћ_default_save_signature
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
-
Йserving_default"
signature_map
!:2dense_10/kernel
 "
trackable_list_wrapper
'
&0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
М
ћlayer_metrics
Ћlayers
 ќlayer_regularization_losses
'regularization_losses
(	variables
)trainable_variables
Ќnon_trainable_variables
ўmetrics
Ќ__call__
┐activity_regularizer_fn
+ќ&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
!:2dense_11/kernel
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
М
Ўlayer_metrics
џlayers
 Џlayer_regularization_losses
,regularization_losses
-	variables
.trainable_variables
юnon_trainable_variables
Юmetrics
Ў__call__
┴activity_regularizer_fn
+ў&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ъlayer_metrics
Ъlayers
 аlayer_regularization_losses
0regularization_losses
1	variables
2trainable_variables
Аnon_trainable_variables
бmetrics
Џ__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
!:2dense_12/kernel
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
'
40"
trackable_list_wrapper
М
Бlayer_metrics
цlayers
 Цlayer_regularization_losses
5regularization_losses
6	variables
7trainable_variables
дnon_trainable_variables
Дmetrics
Ю__call__
├activity_regularizer_fn
+ю&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
еlayer_metrics
Еlayers
 фlayer_regularization_losses
9regularization_losses
:	variables
;trainable_variables
Фnon_trainable_variables
гmetrics
Ъ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
!:2dense_13/kernel
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
М
Гlayer_metrics
«layers
 »layer_regularization_losses
>regularization_losses
?	variables
@trainable_variables
░non_trainable_variables
▒metrics
А__call__
┼activity_regularizer_fn
+а&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
▓layer_metrics
│layers
 ┤layer_regularization_losses
Bregularization_losses
C	variables
Dtrainable_variables
хnon_trainable_variables
Хmetrics
Б__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
#:!2net_output/kernel
 "
trackable_list_wrapper
'
F0"
trackable_list_wrapper
'
F0"
trackable_list_wrapper
М
иlayer_metrics
Иlayers
 ╣layer_regularization_losses
Gregularization_losses
H	variables
Itrainable_variables
║non_trainable_variables
╗metrics
Ц__call__
Кactivity_regularizer_fn
+ц&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
!:2dense_14/kernel
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
М
╝layer_metrics
йlayers
 Йlayer_regularization_losses
Oregularization_losses
P	variables
Qtrainable_variables
┐non_trainable_variables
└metrics
Д__call__
╔activity_regularizer_fn
+д&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
┴layer_metrics
┬layers
 ├layer_regularization_losses
Sregularization_losses
T	variables
Utrainable_variables
─non_trainable_variables
┼metrics
Е__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
!:2dense_15/kernel
 "
trackable_list_wrapper
'
W0"
trackable_list_wrapper
'
W0"
trackable_list_wrapper
М
кlayer_metrics
Кlayers
 ╚layer_regularization_losses
Xregularization_losses
Y	variables
Ztrainable_variables
╔non_trainable_variables
╩metrics
Ф__call__
╦activity_regularizer_fn
+ф&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
╦layer_metrics
╠layers
 ═layer_regularization_losses
\regularization_losses
]	variables
^trainable_variables
╬non_trainable_variables
¤metrics
Г__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
!:2dense_16/kernel
 "
trackable_list_wrapper
'
`0"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
М
лlayer_metrics
Лlayers
 мlayer_regularization_losses
aregularization_losses
b	variables
ctrainable_variables
Мnon_trainable_variables
нmetrics
»__call__
═activity_regularizer_fn
+«&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Нlayer_metrics
оlayers
 Оlayer_regularization_losses
eregularization_losses
f	variables
gtrainable_variables
пnon_trainable_variables
┘metrics
▒__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
!:2dense_17/kernel
 "
trackable_list_wrapper
'
i0"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
М
┌layer_metrics
█layers
 ▄layer_regularization_losses
jregularization_losses
k	variables
ltrainable_variables
Пnon_trainable_variables
яmetrics
│__call__
¤activity_regularizer_fn
+▓&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
▀layer_metrics
Яlayers
 рlayer_regularization_losses
nregularization_losses
o	variables
ptrainable_variables
Рnon_trainable_variables
сmetrics
х__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
!:2dense_18/kernel
 "
trackable_list_wrapper
'
r0"
trackable_list_wrapper
'
r0"
trackable_list_wrapper
М
Сlayer_metrics
тlayers
 Тlayer_regularization_losses
sregularization_losses
t	variables
utrainable_variables
уnon_trainable_variables
Уmetrics
и__call__
Лactivity_regularizer_fn
+Х&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
жlayer_metrics
Жlayers
 вlayer_regularization_losses
wregularization_losses
x	variables
ytrainable_variables
Вnon_trainable_variables
ьmetrics
╣__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
!:2dense_19/kernel
 "
trackable_list_wrapper
'
{0"
trackable_list_wrapper
'
{0"
trackable_list_wrapper
М
Ьlayer_metrics
№layers
 ­layer_regularization_losses
|regularization_losses
}	variables
~trainable_variables
ыnon_trainable_variables
Ыmetrics
╗__call__
Мactivity_regularizer_fn
+║&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
зlayer_metrics
Зlayers
 шlayer_regularization_losses
єregularization_losses
Є	variables
ѕtrainable_variables
Шnon_trainable_variables
эmetrics
й__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
є
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Э0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

щtotal

Щcount
ч	variables
Ч	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
щ0
Щ1"
trackable_list_wrapper
.
ч	variables"
_generic_user_object
&:$2Adam/dense_10/kernel/m
&:$2Adam/dense_11/kernel/m
&:$2Adam/dense_12/kernel/m
&:$2Adam/dense_13/kernel/m
(:&2Adam/net_output/kernel/m
&:$2Adam/dense_14/kernel/m
&:$2Adam/dense_15/kernel/m
&:$2Adam/dense_16/kernel/m
&:$2Adam/dense_17/kernel/m
&:$2Adam/dense_18/kernel/m
&:$2Adam/dense_19/kernel/m
&:$2Adam/dense_10/kernel/v
&:$2Adam/dense_11/kernel/v
&:$2Adam/dense_12/kernel/v
&:$2Adam/dense_13/kernel/v
(:&2Adam/net_output/kernel/v
&:$2Adam/dense_14/kernel/v
&:$2Adam/dense_15/kernel/v
&:$2Adam/dense_16/kernel/v
&:$2Adam/dense_17/kernel/v
&:$2Adam/dense_18/kernel/v
&:$2Adam/dense_19/kernel/v
м2¤
A__inference_model_2_layer_call_and_return_conditional_losses_6082
A__inference_model_2_layer_call_and_return_conditional_losses_6353
A__inference_model_2_layer_call_and_return_conditional_losses_5670
A__inference_model_2_layer_call_and_return_conditional_losses_5832└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩BК
__inference__wrapped_model_4341input_2"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Т2с
&__inference_model_2_layer_call_fn_4840
&__inference_model_2_layer_call_fn_6392
&__inference_model_2_layer_call_fn_6431
&__inference_model_2_layer_call_fn_5508└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_dense_10_layer_call_and_return_all_conditional_losses_6440б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_10_layer_call_fn_6447б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_11_layer_call_and_return_all_conditional_losses_6456б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_11_layer_call_fn_6463б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
─2┴
C__inference_dropout_8_layer_call_and_return_conditional_losses_6468
C__inference_dropout_8_layer_call_and_return_conditional_losses_6480┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ј2І
(__inference_dropout_8_layer_call_fn_6485
(__inference_dropout_8_layer_call_fn_6490┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_dense_12_layer_call_and_return_all_conditional_losses_6499б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_12_layer_call_fn_6506б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
─2┴
C__inference_dropout_9_layer_call_and_return_conditional_losses_6511
C__inference_dropout_9_layer_call_and_return_conditional_losses_6523┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ј2І
(__inference_dropout_9_layer_call_fn_6528
(__inference_dropout_9_layer_call_fn_6533┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_dense_13_layer_call_and_return_all_conditional_losses_6542б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_13_layer_call_fn_6549б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к2├
D__inference_dropout_10_layer_call_and_return_conditional_losses_6554
D__inference_dropout_10_layer_call_and_return_conditional_losses_6566┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
љ2Ї
)__inference_dropout_10_layer_call_fn_6571
)__inference_dropout_10_layer_call_fn_6576┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
H__inference_net_output_layer_call_and_return_all_conditional_losses_6585б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_net_output_layer_call_fn_6592б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_14_layer_call_and_return_all_conditional_losses_6601б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_14_layer_call_fn_6608б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к2├
D__inference_dropout_11_layer_call_and_return_conditional_losses_6613
D__inference_dropout_11_layer_call_and_return_conditional_losses_6625┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
љ2Ї
)__inference_dropout_11_layer_call_fn_6630
)__inference_dropout_11_layer_call_fn_6635┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_dense_15_layer_call_and_return_all_conditional_losses_6644б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_15_layer_call_fn_6651б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к2├
D__inference_dropout_12_layer_call_and_return_conditional_losses_6656
D__inference_dropout_12_layer_call_and_return_conditional_losses_6668┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
љ2Ї
)__inference_dropout_12_layer_call_fn_6673
)__inference_dropout_12_layer_call_fn_6678┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_dense_16_layer_call_and_return_all_conditional_losses_6687б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_16_layer_call_fn_6694б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к2├
D__inference_dropout_13_layer_call_and_return_conditional_losses_6699
D__inference_dropout_13_layer_call_and_return_conditional_losses_6711┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
љ2Ї
)__inference_dropout_13_layer_call_fn_6716
)__inference_dropout_13_layer_call_fn_6721┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_dense_17_layer_call_and_return_all_conditional_losses_6730б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_17_layer_call_fn_6737б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к2├
D__inference_dropout_14_layer_call_and_return_conditional_losses_6742
D__inference_dropout_14_layer_call_and_return_conditional_losses_6754┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
љ2Ї
)__inference_dropout_14_layer_call_fn_6759
)__inference_dropout_14_layer_call_fn_6764┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_dense_18_layer_call_and_return_all_conditional_losses_6773б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_18_layer_call_fn_6780б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к2├
D__inference_dropout_15_layer_call_and_return_conditional_losses_6785
D__inference_dropout_15_layer_call_and_return_conditional_losses_6797┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
љ2Ї
)__inference_dropout_15_layer_call_fn_6802
)__inference_dropout_15_layer_call_fn_6807┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_dense_19_layer_call_and_return_all_conditional_losses_6816б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_19_layer_call_fn_6823б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_add_loss_1_layer_call_and_return_conditional_losses_6828б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_add_loss_1_layer_call_fn_6834б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╔Bк
"__inference_signature_wrapper_5867input_2"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
.__inference_dense_10_activity_regularizer_4354Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_10_layer_call_and_return_conditional_losses_6842б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
.__inference_dense_11_activity_regularizer_4367Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_11_layer_call_and_return_conditional_losses_6850б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
.__inference_dense_12_activity_regularizer_4380Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_12_layer_call_and_return_conditional_losses_6858б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
.__inference_dense_13_activity_regularizer_4393Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_13_layer_call_and_return_conditional_losses_6866б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
р2я
0__inference_net_output_activity_regularizer_4406Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
Ь2в
D__inference_net_output_layer_call_and_return_conditional_losses_6874б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
.__inference_dense_14_activity_regularizer_4419Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_14_layer_call_and_return_conditional_losses_6882б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
.__inference_dense_15_activity_regularizer_4432Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_15_layer_call_and_return_conditional_losses_6890б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
.__inference_dense_16_activity_regularizer_4445Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_16_layer_call_and_return_conditional_losses_6898б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
.__inference_dense_17_activity_regularizer_4458Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_17_layer_call_and_return_conditional_losses_6906б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
.__inference_dense_18_activity_regularizer_4471Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_18_layer_call_and_return_conditional_losses_6914б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
.__inference_dense_19_activity_regularizer_4484Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_19_layer_call_and_return_conditional_losses_6922б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 г
__inference__wrapped_model_4341ѕ&+4=FNW`ir{0б-
&б#
!і
input_2         
ф "GфD
B
tf.math.reduce_sum_1*і'
tf.math.reduce_sum_1         ї
D__inference_add_loss_1_layer_call_and_return_conditional_losses_6828Dб
б
і
inputs 
ф ""б

і
0 
џ
і	
1/0 V
)__inference_add_loss_1_layer_call_fn_6834)б
б
і
inputs 
ф "і X
.__inference_dense_10_activity_regularizer_4354&б
б
і	
x
ф "і │
F__inference_dense_10_layer_call_and_return_all_conditional_losses_6440i&/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_10_layer_call_and_return_conditional_losses_6842[&/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_10_layer_call_fn_6447N&/б,
%б"
 і
inputs         
ф "і         X
.__inference_dense_11_activity_regularizer_4367&б
б
і	
x
ф "і │
F__inference_dense_11_layer_call_and_return_all_conditional_losses_6456i+/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_11_layer_call_and_return_conditional_losses_6850[+/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_11_layer_call_fn_6463N+/б,
%б"
 і
inputs         
ф "і         X
.__inference_dense_12_activity_regularizer_4380&б
б
і	
x
ф "і │
F__inference_dense_12_layer_call_and_return_all_conditional_losses_6499i4/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_12_layer_call_and_return_conditional_losses_6858[4/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_12_layer_call_fn_6506N4/б,
%б"
 і
inputs         
ф "і         X
.__inference_dense_13_activity_regularizer_4393&б
б
і	
x
ф "і │
F__inference_dense_13_layer_call_and_return_all_conditional_losses_6542i=/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_13_layer_call_and_return_conditional_losses_6866[=/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_13_layer_call_fn_6549N=/б,
%б"
 і
inputs         
ф "і         X
.__inference_dense_14_activity_regularizer_4419&б
б
і	
x
ф "і │
F__inference_dense_14_layer_call_and_return_all_conditional_losses_6601iN/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_14_layer_call_and_return_conditional_losses_6882[N/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_14_layer_call_fn_6608NN/б,
%б"
 і
inputs         
ф "і         X
.__inference_dense_15_activity_regularizer_4432&б
б
і	
x
ф "і │
F__inference_dense_15_layer_call_and_return_all_conditional_losses_6644iW/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_15_layer_call_and_return_conditional_losses_6890[W/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_15_layer_call_fn_6651NW/б,
%б"
 і
inputs         
ф "і         X
.__inference_dense_16_activity_regularizer_4445&б
б
і	
x
ф "і │
F__inference_dense_16_layer_call_and_return_all_conditional_losses_6687i`/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_16_layer_call_and_return_conditional_losses_6898[`/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_16_layer_call_fn_6694N`/б,
%б"
 і
inputs         
ф "і         X
.__inference_dense_17_activity_regularizer_4458&б
б
і	
x
ф "і │
F__inference_dense_17_layer_call_and_return_all_conditional_losses_6730ii/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_17_layer_call_and_return_conditional_losses_6906[i/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_17_layer_call_fn_6737Ni/б,
%б"
 і
inputs         
ф "і         X
.__inference_dense_18_activity_regularizer_4471&б
б
і	
x
ф "і │
F__inference_dense_18_layer_call_and_return_all_conditional_losses_6773ir/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_18_layer_call_and_return_conditional_losses_6914[r/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_18_layer_call_fn_6780Nr/б,
%б"
 і
inputs         
ф "і         X
.__inference_dense_19_activity_regularizer_4484&б
б
і	
x
ф "і │
F__inference_dense_19_layer_call_and_return_all_conditional_losses_6816i{/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_19_layer_call_and_return_conditional_losses_6922[{/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_19_layer_call_fn_6823N{/б,
%б"
 і
inputs         
ф "і         ц
D__inference_dropout_10_layer_call_and_return_conditional_losses_6554\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ ц
D__inference_dropout_10_layer_call_and_return_conditional_losses_6566\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ |
)__inference_dropout_10_layer_call_fn_6571O3б0
)б&
 і
inputs         
p 
ф "і         |
)__inference_dropout_10_layer_call_fn_6576O3б0
)б&
 і
inputs         
p
ф "і         ц
D__inference_dropout_11_layer_call_and_return_conditional_losses_6613\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ ц
D__inference_dropout_11_layer_call_and_return_conditional_losses_6625\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ |
)__inference_dropout_11_layer_call_fn_6630O3б0
)б&
 і
inputs         
p 
ф "і         |
)__inference_dropout_11_layer_call_fn_6635O3б0
)б&
 і
inputs         
p
ф "і         ц
D__inference_dropout_12_layer_call_and_return_conditional_losses_6656\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ ц
D__inference_dropout_12_layer_call_and_return_conditional_losses_6668\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ |
)__inference_dropout_12_layer_call_fn_6673O3б0
)б&
 і
inputs         
p 
ф "і         |
)__inference_dropout_12_layer_call_fn_6678O3б0
)б&
 і
inputs         
p
ф "і         ц
D__inference_dropout_13_layer_call_and_return_conditional_losses_6699\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ ц
D__inference_dropout_13_layer_call_and_return_conditional_losses_6711\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ |
)__inference_dropout_13_layer_call_fn_6716O3б0
)б&
 і
inputs         
p 
ф "і         |
)__inference_dropout_13_layer_call_fn_6721O3б0
)б&
 і
inputs         
p
ф "і         ц
D__inference_dropout_14_layer_call_and_return_conditional_losses_6742\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ ц
D__inference_dropout_14_layer_call_and_return_conditional_losses_6754\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ |
)__inference_dropout_14_layer_call_fn_6759O3б0
)б&
 і
inputs         
p 
ф "і         |
)__inference_dropout_14_layer_call_fn_6764O3б0
)б&
 і
inputs         
p
ф "і         ц
D__inference_dropout_15_layer_call_and_return_conditional_losses_6785\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ ц
D__inference_dropout_15_layer_call_and_return_conditional_losses_6797\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ |
)__inference_dropout_15_layer_call_fn_6802O3б0
)б&
 і
inputs         
p 
ф "і         |
)__inference_dropout_15_layer_call_fn_6807O3б0
)б&
 і
inputs         
p
ф "і         Б
C__inference_dropout_8_layer_call_and_return_conditional_losses_6468\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ Б
C__inference_dropout_8_layer_call_and_return_conditional_losses_6480\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ {
(__inference_dropout_8_layer_call_fn_6485O3б0
)б&
 і
inputs         
p 
ф "і         {
(__inference_dropout_8_layer_call_fn_6490O3б0
)б&
 і
inputs         
p
ф "і         Б
C__inference_dropout_9_layer_call_and_return_conditional_losses_6511\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ Б
C__inference_dropout_9_layer_call_and_return_conditional_losses_6523\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ {
(__inference_dropout_9_layer_call_fn_6528O3б0
)б&
 і
inputs         
p 
ф "і         {
(__inference_dropout_9_layer_call_fn_6533O3б0
)б&
 і
inputs         
p
ф "і         я
A__inference_model_2_layer_call_and_return_conditional_losses_5670ў&+4=FNW`ir{8б5
.б+
!і
input_2         
p 

 
ф "╬б╩
і
0         
«џф
і	
1/0 
і	
1/1 
і	
1/2 
і	
1/3 
і	
1/4 
і	
1/5 
і	
1/6 
і	
1/7 
і	
1/8 
і	
1/9 
і

1/10 
і

1/11 я
A__inference_model_2_layer_call_and_return_conditional_losses_5832ў&+4=FNW`ir{8б5
.б+
!і
input_2         
p

 
ф "╬б╩
і
0         
«џф
і	
1/0 
і	
1/1 
і	
1/2 
і	
1/3 
і	
1/4 
і	
1/5 
і	
1/6 
і	
1/7 
і	
1/8 
і	
1/9 
і

1/10 
і

1/11 П
A__inference_model_2_layer_call_and_return_conditional_losses_6082Ќ&+4=FNW`ir{7б4
-б*
 і
inputs         
p 

 
ф "╬б╩
і
0         
«џф
і	
1/0 
і	
1/1 
і	
1/2 
і	
1/3 
і	
1/4 
і	
1/5 
і	
1/6 
і	
1/7 
і	
1/8 
і	
1/9 
і

1/10 
і

1/11 П
A__inference_model_2_layer_call_and_return_conditional_losses_6353Ќ&+4=FNW`ir{7б4
-б*
 і
inputs         
p

 
ф "╬б╩
і
0         
«џф
і	
1/0 
і	
1/1 
і	
1/2 
і	
1/3 
і	
1/4 
і	
1/5 
і	
1/6 
і	
1/7 
і	
1/8 
і	
1/9 
і

1/10 
і

1/11 Є
&__inference_model_2_layer_call_fn_4840]&+4=FNW`ir{8б5
.б+
!і
input_2         
p 

 
ф "і         Є
&__inference_model_2_layer_call_fn_5508]&+4=FNW`ir{8б5
.б+
!і
input_2         
p

 
ф "і         є
&__inference_model_2_layer_call_fn_6392\&+4=FNW`ir{7б4
-б*
 і
inputs         
p 

 
ф "і         є
&__inference_model_2_layer_call_fn_6431\&+4=FNW`ir{7б4
-б*
 і
inputs         
p

 
ф "і         Z
0__inference_net_output_activity_regularizer_4406&б
б
і	
x
ф "і х
H__inference_net_output_layer_call_and_return_all_conditional_losses_6585iF/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 Б
D__inference_net_output_layer_call_and_return_conditional_losses_6874[F/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ {
)__inference_net_output_layer_call_fn_6592NF/б,
%б"
 і
inputs         
ф "і         ║
"__inference_signature_wrapper_5867Њ&+4=FNW`ir{;б8
б 
1ф.
,
input_2!і
input_2         "GфD
B
tf.math.reduce_sum_1*і'
tf.math.reduce_sum_1         