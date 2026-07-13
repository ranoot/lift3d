#include "instance_ids.h"
#include "inf_client.h"        // FrameResult

#include <cstddef>
#include <unordered_set>

int InstanceIdGraph::find(int x) {
    auto it = parent_.find(x);
    if (it == parent_.end()) { parent_[x] = x; return x; }   // lazy singleton
    int root = x;
    while (parent_[root] != root) root = parent_[root];
    while (parent_[x] != root) { int nx = parent_[x]; parent_[x] = root; x = nx; }
    return root;
}

void InstanceIdGraph::unite(int a, int b) {
    const int ra = find(a), rb = find(b);
    if (ra != rb) parent_[ra] = rb;
}

int InstanceIdGraph::root(int id) { return find(id); }

void InstanceIdGraph::ingestFrame(const FrameResult& fr,
                                  const std::function<bool(int)>& is_thing_label) {
    const int H = fr.h, W = fr.w;
    const std::size_t n = (std::size_t)H * W;
    if (H <= 0 || W <= 0 || fr.id_map.size() != n || fr.label_map.size() != n) return;

    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            const std::size_t o = (std::size_t)v * W + u;
            const int a  = fr.id_map[o];
            if (a < 0) continue;
            const int la = fr.label_map[o];
            if (!is_thing_label(la)) continue;             // only bridge thing instances
            if (u + 1 < W) {                               // right neighbour
                const std::size_t o2 = o + 1;
                const int b = fr.id_map[o2];
                if (b >= 0 && b != a && fr.label_map[o2] == la) unite(a, b);
            }
            if (v + 1 < H) {                               // down neighbour
                const std::size_t o2 = o + (std::size_t)W;
                const int b = fr.id_map[o2];
                if (b >= 0 && b != a && fr.label_map[o2] == la) unite(a, b);
            }
        }
    }
}

bool InstanceIdGraph::overlaps(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.empty() || b.empty()) return false;
    std::unordered_set<int> ra;
    ra.reserve(a.size());
    for (int id : a) if (id >= 0) ra.insert(find(id));
    for (int id : b) if (id >= 0 && ra.count(find(id))) return true;
    return false;
}
