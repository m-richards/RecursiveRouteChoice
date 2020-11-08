# Recursive Route Choice
This package implements the recursive logit model for prediction and estimation. [See the docs
 for details.](https://m-richards.github.io/RecursiveRouteChoice/)
The below discussion on data formats will shortly be superceded by a corresponding page in the
 Sphinx documentation.

### Assumptions on Data Formats
Each observation is read in the format
```
[Dest, Orig, Link1, Link2, ..., LinkN, Dest]
```
where there may or may not be zero padding after Dest - see below.
We prefer json formatting without zero padding, but sparse COO format is supported to allow
Tien Mai's repository examples to be used.

As a simple example, consider the list of two observations (sequence from 3 to 1 and from 4 to 2
). Below we discuss the encoding for this simple example.
 ```
[ [1, 3, 2, 6, 8, 9, 1],
   [2, 4, 5, 2, 0, 0, 0]] 
 ```
##### Some quirks
- Destination can appear more than once in this format. This is because the "true destination" in
 terms of the model is an internally appended dummy sink arc. So one can visit the destination
  arc and not travel on to the destination node.

#### JSON format (preferred):
```
[[0, 2, 1, 5, 7, 8, 0], [1, 3, 4, 1]] 
```
- Note entry are ragged, there is no zero padding 
- Note the zero based indexing(one based indexing \*should\* work), since we no longer need to
 reserve zero for sparsity
- Any valid json representation (with arbitrary indentation) will suffice.
- Internally we read the json into Awkward Arrays to allow convenient numpy like indexing and
 function applications, that one wouldn't have from a list of lists.

#### COO format:

these are stored in a sparse format with a single entry per line.

This sequence of two observations is encoded as 
```
1 1 1
1 2 3
1 3 2
1 4 6
1 5 8
1 6 9
1 7 1
2 1 2
2 2 4
2 3 5
2 4 2
```
Note that this is one based indexing - the code converts, since the Tien Mai compatible data was
 a matlab code.