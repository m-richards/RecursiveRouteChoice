Data Format Support
===================

Assumptions on Observation Data Formats
---------------------------------------

Each observation is read in the format

::

    [Dest, Orig, Link1, Link2, ..., LinkN, Dest]

where there may or may not be zero padding after Dest - see below. This naturally leads to the
collection of observations being stored as a list of lists. We prefer json formatting without zero padding, but sparse COO format is supported to allow
Tien Mai's repository examples to be used.

As a simple example, consider the list of two observations (sequence from 3 to 1 and from 4 to 2
). Below we discuss the encoding for this simple example.

::

    [ [1, 3, 2, 6, 8, 9, 1],
      [2, 4, 5, 2, 0, 0, 0]]

Some quirks
^^^^^^^^^^^

- The destination can appear more than once in this format. This is because the "true
  destination" in  terms of the model is an internally appended dummy sink arc. So one can visit
  the destination arc and not travel on to the destination node.

- Reasonably this would not be observed, but it can be predicted if the model is not well specified


JSON format (preferred):
------------------------

::

    [
        [0, 2, 1, 5, 7, 8, 0],
        [1, 3, 4, 1]
    ]


- Note entries are ragged, there is no zero padding
- Note the zero based indexing, that is "0" is a legal node index. This is not possible with
  native COO format as "0" is a dummy value to pad out the matrix. (one based indexing
  *should* work as well, as there is no special significance of zero
- Any valid json representation (with arbitrary indentation) will suffice.
- Internally we read the json into Awkward Arrays to allow convenient numpy like indexing and
  function applications, that one wouldn't have from a list of lists.

COO format:
-----------

The observations are stored in a sparse format with a single entry per line.
The sequence of two observations above is encoded as

::

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

Note that this is one based indexing since the Tien Mai data was originally used for
matlab code. This format is confusing to read, and it is not particularly efficient as the
entries are known to have a block sparsity pattern that is left justified, so the indexing is
somewhat excessive. Also, we have added 1 to all the node indexes, as in this format zero is fill
value. Internally, data is immediately converted out of this format into awkward arrays.