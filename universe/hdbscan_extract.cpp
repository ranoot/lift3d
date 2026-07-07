#include "hdbscan_extract.h"

#include <algorithm>
#include <limits>
#include <queue>

// Standard HDBSCAN* flat extraction (condense tree -> stability -> EOM select ->
// label). Node ids: leaves [0, n), internal [n, n+L) where internal id (n+i) is
// linkage[i]; the root is the last row. Condensed-cluster ids start at n (the root
// cluster) and grow; we index per-cluster arrays by (id - n).

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Leaves under a node id (1 for a point leaf, else the row's `size`).
inline long leafCount(const std::vector<LinkNode>& lk, int n, long node) {
    return node < n ? 1L : lk[(std::size_t)(node - n)].size;
}

// BFS over the FULL hierarchy from `start`, returning every reachable node id
// (start first). Children of an internal node are its two linkage members.
std::vector<long> bfsHierarchy(const std::vector<LinkNode>& lk, int n, long start) {
    std::vector<long> order;
    std::queue<long> q;
    q.push(start);
    while (!q.empty()) {
        long node = q.front(); q.pop();
        order.push_back(node);
        if (node >= n) {
            const LinkNode& r = lk[(std::size_t)(node - n)];
            q.push(r.a);
            q.push(r.b);
        }
    }
    return order;
}

// Union-find used only for labelling (attach the larger id under the smaller, so
// find() climbs toward the root / nearest un-unioned ancestor).
struct UF {
    std::vector<long> p;
    explicit UF(long m) : p(m) { for (long i = 0; i < m; ++i) p[i] = i; }
    long find(long x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void unite(long child, long parent) {         // merge child's set into parent's
        long a = find(child), b = find(parent);
        if (a == b) return;
        if (a < b) std::swap(a, b);               // b is the smaller (closer-to-root) id
        p[a] = b;
    }
};

}  // namespace

std::vector<int> extractFlatClusters(const std::vector<LinkNode>& linkage, int n,
                                     int min_cluster_size, bool allow_single_cluster) {
    std::vector<int> labels((std::size_t)std::max(0, n), -1);
    if (n <= 0 || linkage.empty()) return labels;
    if (min_cluster_size < 2) min_cluster_size = 2;

    const int  L    = (int)linkage.size();       // = n-1 for a connected MST
    const long total = (long)n + L;              // total node ids
    const long root  = total - 1;                // last row's node id
    const long mcs   = min_cluster_size;

    // ---- 1. Condense the tree ------------------------------------------------
    // relabel[node] = condensed cluster id the node belongs to (root cluster = n).
    std::vector<long> relabel((std::size_t)total, -1);
    std::vector<char> ignore((std::size_t)total, 0);
    relabel[root] = n;
    long next_label = n + 1;

    struct CRow { long parent, child; double lambda; long size; };
    std::vector<CRow> cond;
    cond.reserve((std::size_t)L * 2);

    for (long node : bfsHierarchy(linkage, n, root)) {
        if (node < n || ignore[node]) continue;
        const LinkNode& r = linkage[(std::size_t)(node - n)];
        const long   left = r.a, right = r.b;
        const double lambda = r.height > 0.0 ? 1.0 / r.height : kInf;
        const long   lc = leafCount(linkage, n, left);
        const long   rc = leafCount(linkage, n, right);

        auto fallOut = [&](long sub_root) {        // all leaves under sub_root -> noise
            for (long sub : bfsHierarchy(linkage, n, sub_root)) {
                if (sub < n) cond.push_back({relabel[node], sub, lambda, 1});
                ignore[sub] = 1;
            }
        };

        if (lc >= mcs && rc >= mcs) {              // genuine split into two clusters
            relabel[left]  = next_label++;
            cond.push_back({relabel[node], relabel[left], lambda, lc});
            relabel[right] = next_label++;
            cond.push_back({relabel[node], relabel[right], lambda, rc});
        } else if (lc < mcs && rc < mcs) {         // whole node dissolves into noise
            fallOut(node);
        } else if (lc < mcs) {                     // right stays this cluster, left sheds
            relabel[right] = relabel[node];
            fallOut(left);
        } else {                                   // left stays this cluster, right sheds
            relabel[left] = relabel[node];
            fallOut(right);
        }
    }

    const long ncl = next_label - n;               // number of condensed clusters
    auto cidx = [&](long cluster) { return (std::size_t)(cluster - n); };

    // ---- 2. Stability (excess of mass) --------------------------------------
    std::vector<double> births((std::size_t)ncl, kInf);
    births[0] = 0.0;                               // root cluster born at lambda 0
    for (const CRow& c : cond)
        if (c.child >= n) births[cidx(c.child)] = c.lambda;   // child cluster's birth
    std::vector<double> stability((std::size_t)ncl, 0.0);
    for (const CRow& c : cond)
        stability[cidx(c.parent)] += (c.lambda - births[cidx(c.parent)]) * (double)c.size;

    // ---- 3. EOM cluster selection -------------------------------------------
    // Bottom-up (descending id): keep a cluster iff it is at least as stable as the
    // sum of its selected descendants; otherwise defer to the descendants.
    std::vector<std::vector<long>> childClusters((std::size_t)ncl);
    for (const CRow& c : cond)
        if (c.child >= n) childClusters[cidx(c.parent)].push_back(c.child);

    std::vector<char> is_cluster((std::size_t)ncl, 1);
    for (long cluster = next_label - 1; cluster >= n; --cluster) {
        if (!allow_single_cluster && cluster == n) break;   // never select the root
        const std::size_t k = cidx(cluster);
        double sub = 0.0;
        for (long ch : childClusters[k]) sub += stability[cidx(ch)];
        if (sub > stability[k]) {
            is_cluster[k] = 0;
            stability[k] = sub;                    // propagate accumulated stability up
        } else {                                   // select `cluster`, drop its descendants
            std::queue<long> q;
            for (long ch : childClusters[k]) q.push(ch);
            while (!q.empty()) {
                long d = q.front(); q.pop();
                is_cluster[cidx(d)] = 0;
                for (long ch : childClusters[cidx(d)]) q.push(ch);
            }
        }
    }
    if (!allow_single_cluster) is_cluster[0] = 0;

    // Compact labels 0..K-1 for the selected clusters (ascending id).
    std::vector<int> compact((std::size_t)ncl, -1);
    int K = 0;
    for (long cluster = n; cluster < next_label; ++cluster)
        if (is_cluster[cidx(cluster)]) compact[cidx(cluster)] = K++;

    // ---- 4. Label points ----------------------------------------------------
    // Collapse every non-selected cluster into its parent, then each point takes the
    // label of its (find-resolved) attaching cluster if that cluster is selected.
    UF uf(next_label);
    for (const CRow& c : cond)
        if (c.child >= n && !is_cluster[cidx(c.child)]) uf.unite(c.child, c.parent);
    for (const CRow& c : cond) {
        if (c.child >= n) continue;                // points only
        const long resolved = uf.find(c.parent);
        if (is_cluster[cidx(resolved)]) labels[(std::size_t)c.child] = compact[cidx(resolved)];
    }
    return labels;
}
