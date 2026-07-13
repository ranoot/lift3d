#pragma once
// InstanceIdGraph: a persistent union-find over OV-DVIS++ instance ids that BRIDGES
// the 2D over-segmentation measured in exp_per_view (one physical object frequently
// carries several stable instance ids). Same-class instance ids whose pixel regions
// physically touch are treated as one object. Instance ids are temporally stable, so
// unions accumulate across frames. Consolidation uses this as an OR-signal alongside
// the semantic class-histogram affinity -- and as a positive identity signal on top of
// it (adjacency alone over-merges; see object_consolidate.cpp).
//
// Tiers store the RAW instance ids they observed (deduped); roots are resolved live in
// overlaps(), so a union discovered on a later frame still bridges proposals that were
// aggregated earlier.

#include <functional>
#include <unordered_map>
#include <vector>

struct FrameResult;   // inf_client.h

class InstanceIdGraph {
public:
    // Ingest one frame's id_map/label_map: union same-class, 4-adjacent (right + down)
    // thing instance ids. is_thing_label(label) decides whether a per-pixel class id is
    // a thing (stuff/background pixels never seed a union).
    void ingestFrame(const FrameResult& fr,
                     const std::function<bool(int)>& is_thing_label);

    // Root of an id's co-touch group (lazily creates a singleton for an unseen id).
    int  root(int id);

    // True if any id in `a` shares a co-touch group with any id in `b`.
    bool overlaps(const std::vector<int>& a, const std::vector<int>& b);

private:
    int  find(int x);
    void unite(int a, int b);
    std::unordered_map<int, int> parent_;   // id -> parent id (union-find forest)
};
