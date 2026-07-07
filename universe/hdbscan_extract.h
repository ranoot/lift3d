#pragma once
// Flat-cluster extraction for HDBSCAN*.
//
// The wangyiqiu/hdbscan library only produces a DENDROGRAM (a SciPy-style linkage:
// the mutual-reachability MST condensed into n-1 merge rows). It gives no flat
// clustering. This module turns that linkage into the strict, noise-aware flat
// clustering HDBSCAN* is known for, using the standard CONDENSED TREE + EXCESS OF
// MASS (EOM) selection of Campello/McInnes. Kept deliberately parlay-free (plain
// std types only) so it is unit-testable on synthetic linkages and does not leak
// the heavy parallel runtime into its includers.
//
// Linkage encoding (matches pargeo::dendrogram, dendrogram.cpp): points are leaves
// 0..n-1; row i creates internal node id n+i by merging children a,b at `height`
// (the merge distance); `size` is the number of leaves under the new node. Rows are
// sorted by ascending height. lambda := 1/height is the density level.

#include <vector>

// One dendrogram/linkage row: merge children `a` and `b` (node ids) at `height`,
// producing a node covering `size` leaves. Mirrors pargeo::dendroNode as plain data.
struct LinkNode {
    long   a = 0, b = 0;   // child node ids (< this row's own id)
    double height = 0.0;   // merge distance
    long   size = 0;       // # leaves under the merged node
};

// Flat cluster label per point (0..K-1), or -1 for noise. `n` is the point count;
// `linkage` must hold the n-1 rows in dendrogram order. `min_cluster_size` is the
// strictness knob (clusters smaller than this are dissolved into noise). If
// `allow_single_cluster` is true (default) the whole set may resolve to one cluster
// -- required so a class that is a single object is still detected (rather than
// discarded as noise); set false to forbid the trivial all-in-one clustering.
std::vector<int> extractFlatClusters(const std::vector<LinkNode>& linkage, int n,
                                     int min_cluster_size,
                                     bool allow_single_cluster = true);
