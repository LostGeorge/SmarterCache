# SmarterCache
#### Towards a Generalizable Deep Learning Approach For Cache Replacement
SmarterCache is a deep learning framework to emulate optimal cache replacement
blind to item size. Built using PyTorch, SmarterCache outperforms heuristic
algorithms such a ARC and SLRU, as well as more basic methods such as LRU and
LFU. SmarterCache is not a complete black box; we use the SHAP package to analyze
how each feature affects the model output, allowing sensible implementation. For
comparison to other algorithms, PyMimircache by Yang et al. is employed.

## Getting Started
SmarterCache requires Python 3.5 or higher, and has full functionaliy on Linux.
We provide full compatibiliy for Windows and MacOS for all modules that do not use
PyMimircache.

#### Prerequisites
If you want to install all possible prerequisites, they are available through the
pip package manger for python:
```
pip install numpy pandas matplotlib sklearn shap torch PyMimircache
```
#### Installing
If you're on github we assume that you have git installed. Go to the folder you
desire SmarterCache to be installed in and do:
```
git clone https://github.com/LostHerro/SmarterCache.git
```

## Functionality
#### Generating Features
We require that all traces be in any text format with each row being a request and
the columns separated by a regular expression. To generate the features for a trace, 
add the desired trace into the folder [SmarterCache/feat/traces](SmarterCache/feat/traces).
Peek at the trace using some tool to find which 0-indexed columns have the system time and the
id of the request item, along with how the columns are divided. Now go to 
[gen_features.py](SmarterCache/feat/gen_features.py) and edit line 142 and 145 so that
```
col_names = ['time', 'id'] # if the system time column comes first
col_names = ['id', 'time'] # if the request id column comes first

# For example, regex is ',' for a .csv file or '\t' for a .tsv file.
# Column indices is [0, 4] if the first of time or id is the first column and
# the other is the fifth column.
df = pd.read_csv(pd.read_csv(source_file, sep=regex, usecols=column indices, ...)
```
Then from the [SmarterCache/feat](SmarterCache/feat) folder, run
```
python3 gen_features.py [FILE_NAME_HERE]
```
The generated features for the trace will be saved in the 
[SmarterCache/feat/features](SmarterCache/feat/features) folder.

#### Running SmarterCache
The preset parameters for SmarterCache in regards to the learning system should be
sufficient. If you wish to alter these, go ahead and dig into the
[run_cache.py](SmarterCache/run_cache.py) file.

For other algorithms to compare SmarterCache to, PyMimircache is used. Go to line
317 of the [run_cache.py](SmarterCache/run_cache.py) file, and add whichever algorithms
found in PyMimircache that are desired. By default we have
```
comparison_lst = ['Optimal', 'LRU', 'LFU', 'Random', 'SLRU', 'ARC']
```

The two parameters to run SmarterCache are the normalization constant, and the
maximum cache size (in number of items) desired to be analyzed for the hit ratio
curve. We recommend using normalization constant = maximum cache size, but for
very large or very small maximum cache sizes, it could be better to make the
normalization constant repsectively smaller or greater than the maximum.

Now finally, to actually run SmarterCache, go to the [SmarterCache](SmarterCache)
folder, and do
```
python3 run_cache.py [TRACE_NAME] [NORMALIZATION_CONSTANT] [MAXIMUM_CACHE_SIZE]
```

**_WARNING:_ This code is unoptimized and is hindered by python computation speed.
The training is relatively quick, but doing online cache evaluation is very slow.
Runtimes for workloads of size 1.5 million are about 2 hours, and scale slightly more
than linearly with workload size.**

The hit ratio curve for the cache will the saved in 
[SmarterCache/eval/hrc](SmarterCache/eval/hrc)

#### Model Evaluation
Evaluating SmarterCache's predictions is quite simple. First noting that
SmarterCache's output is the time from the present until future access for a
cache item, the SHAP values plot provides a way to see how various features
values contribute towards the model output. From the [SmarterCache](SmarterCache)
folder, run
```
python3 shap_analysis.py [TRACE_NAME] [NORMALIZATION_CONSTANT]
```
Two SHAP value plots will be saved, one with normal scale and the other with log
scale. One excel file containing statistics for the SHAP values will be generated
as well. These can be found in [SmarterCache/eval/shap](SmarterCache/eval/shap).

## Addendum
To be done later.
