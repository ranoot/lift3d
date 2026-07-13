#include "objects.h"
#include "voxel_util.h"         // objutil::VKey/VHash/VSet, voxelOf, cosine
#include "instance_ids.h"       // InstanceIdGraph::overlaps (2D co-touch id bonus)

// The proposals -> objects layer (tier 2 -> tier 3), as a single-tier Union-Find over a
// persistent COMPONENT layer. Parlay-free: reads only Universe / ObjectSeeds + std. Each
// per-frame proposal is matched, by voxel overlap, against the components that already own
// its voxels; it joins (and progressively unions) every same-class component whose
// CONTAINMENT clears that component's level-indexed bar, contributing only its NEW (unowned)
// voxels and one unit of support. An "object" is virtual: a component whose support has
// reached min_merges, synthesised from its aggregated voxel footprint. Geometry is stored
// ONCE per component (deduped voxel -> representative gidx), not per observation, so memory
// is bounded by the scene, not the run length; and only components touched can change, so
// the per-frame rebuild scans components, never the full observation history.
//
// Two correctness properties this layer must keep:
//   - Fresh-only claim: a proposal contributes ONLY its unowned voxels to the component it
//     joins; voxels already owned by another component are overlap EVIDENCE, never stolen.
//   - True containment: the merge bar is shared / min(|proposal|, |component|), so a large
//     later full-view proposal can still be absorbed by a smaller established object.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using objutil::VKey;
using objutil::VHash;
using objutil::voxelOf;

// -- Union-Find over component ids --------------------------------------------------------

int Objects::find(int c) {
    int r = c;
    while (parent_[r] != r) r = parent_[r];
    while (parent_[c] != r) { const int n = parent_[c]; parent_[c] = r; c = n; }  // compress
    return r;
}

// Merge the component with FEWER voxels into the one with more (cheap), folding its voxel
// map, class votes and support; the drained component's maps are cleared. Returns the root.
int Objects::unite(int a, int b) {
    a = find(a); b = find(b);
    if (a == b) return a;
    if (comp_[a].vox.size() < comp_[b].vox.size()) std::swap(a, b);
    Comp& A = comp_[a];
    Comp& B = comp_[b];
    for (const auto& kv : B.vox) A.vox.emplace(kv.first, kv.second);  // emplace: on a shared voxel (overlap_sets) keep A's representative
    for (const auto& kv : B.cls) A.cls[kv.first] += kv.second;
    for (int id : B.ids)         A.ids.insert(id);
    A.support += B.support;
    B.vox.clear(); B.cls.clear(); B.ids.clear(); B.support = 0;
    parent_[b] = a;
    return a;
}

// -- global_context helpers ---------------------------------------------------------------

// Order-independent packed key for an edge between two component ids.
std::uint64_t Objects::ekey(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32)
         | static_cast<std::uint32_t>(b);
}

// Fraction of a component's voxels that fall in this frame's view set (SAI3D seg_seen);
// 1.0 when no view is supplied (confidence weighting disabled).
float Objects::visRatio(const Comp& c, const objutil::VSet* view) const {
    if (!view || c.vox.empty()) return 1.0f;
    int seen = 0;
    for (const auto& kv : c.vox) if (view->count(kv.first)) ++seen;
    return static_cast<float>(seen) / static_cast<float>(c.vox.size());
}

int Objects::newComp() {
    const int id = static_cast<int>(parent_.size());
    parent_.push_back(id);
    comp_.emplace_back();
    return id;
}

// -- Virtual object synthesis -------------------------------------------------------------

void Objects::rebuildCache(const Universe& uni, const Params& g) {
    cache_.clear();
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    if (!cloud || cloud->empty()) return;

    // overlap_sets: components may share voxels. Resolve each voxel to the promoted component
    // with the HIGHEST support (tie: larger voxel set, then lower id) so the synthesised
    // objects are a clean disjoint partition even though the internal sets overlap.
    if (g.overlap_sets) {
        std::unordered_map<VKey, int, VHash> win;             // voxel -> winning root id
        auto better = [&](int a, int b) -> bool {             // is root a a stronger owner than b?
            const int sa = comp_[a].support, sb = comp_[b].support;
            if (sa != sb) return sa > sb;
            const std::size_t va = comp_[a].vox.size(), vb = comp_[b].vox.size();
            if (va != vb) return va > vb;
            return a < b;
        };
        for (int id = 0; id < static_cast<int>(comp_.size()); ++id) {
            if (parent_[id] != id || comp_[id].support < g.min_merges) continue;
            for (const auto& kv : comp_[id].vox) {
                auto it = win.find(kv.first);
                if (it == win.end())        win.emplace(kv.first, id);
                else if (better(id, it->second)) it->second = id;
            }
        }
        for (int id = 0; id < static_cast<int>(comp_.size()); ++id) {
            if (parent_[id] != id) continue;
            const Comp& c = comp_[id];
            if (c.support < g.min_merges) continue;
            int best_c = -1, best_n = -1;
            for (const auto& kv : c.cls) if (kv.second > best_n) { best_n = kv.second; best_c = kv.first; }
            Object o;
            o.id       = id;
            o.class_id = best_c;
            o.kind     = best_c >= 0 ? uni.semantics().kind(best_c) : ClassKind::Thing;
            o.level    = c.support;
            double s[3] = {0, 0, 0};
            o.points.reserve(c.vox.size());
            for (const auto& kv : c.vox) {
                auto w = win.find(kv.first);
                if (w == win.end() || w->second != id) continue;   // a stronger component won it
                const int gid = kv.second;
                if (gid < 0 || gid >= static_cast<int>(cloud->size())) continue;
                if (!uni.pointAlive(gid)) continue;
                const Universe::PointT& p = (*cloud)[gid];
                s[0] += p.x; s[1] += p.y; s[2] += p.z;
                o.points.push_back(gid);
            }
            if (o.points.empty()) continue;
            const double n = static_cast<double>(o.points.size());
            o.centroid[0] = static_cast<float>(s[0] / n);
            o.centroid[1] = static_cast<float>(s[1] / n);
            o.centroid[2] = static_cast<float>(s[2] / n);
            cache_.push_back(std::move(o));
        }
        return;
    }

    for (int id = 0; id < static_cast<int>(comp_.size()); ++id) {
        if (parent_[id] != id) continue;                      // not a root (absorbed component)
        const Comp& c = comp_[id];
        if (c.support < g.min_merges) continue;               // not promoted yet -> hidden

        int best_c = -1, best_n = -1;                         // majority component class
        for (const auto& kv : c.cls) if (kv.second > best_n) { best_n = kv.second; best_c = kv.first; }

        Object o;
        o.id       = id;
        o.class_id = best_c;
        o.kind     = best_c >= 0 ? uni.semantics().kind(best_c) : ClassKind::Thing;
        o.level    = c.support;
        double s[3] = {0, 0, 0};
        o.points.reserve(c.vox.size());
        for (const auto& kv : c.vox) {
            const int gid = kv.second;
            if (gid < 0 || gid >= static_cast<int>(cloud->size())) continue;
            if (!uni.pointAlive(gid)) continue;               // tombstoned (dynamic) -> skip
            const Universe::PointT& p = (*cloud)[gid];
            s[0] += p.x; s[1] += p.y; s[2] += p.z;
            o.points.push_back(gid);
        }
        if (o.points.empty()) continue;
        const double n = static_cast<double>(o.points.size());
        o.centroid[0] = static_cast<float>(s[0] / n);
        o.centroid[1] = static_cast<float>(s[1] / n);
        o.centroid[2] = static_cast<float>(s[2] / n);
        cache_.push_back(std::move(o));
    }
}

// -- Per-frame fold-in --------------------------------------------------------------------

void Objects::consolidate(const Universe& uni, const ObjectSeeds& seeds, const Params& g,
                          float voxel, const objutil::VSet* view_vox, InstanceIdGraph* idg) {
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    if (!cloud || cloud->empty()) return;
    const std::vector<ObjectSeed>& props = seeds.list();
    if (props.empty()) return;

    const float inv  = 1.0f / (voxel > 0.0f ? voxel : 0.05f);
    const bool  gc   = g.global_context;
    const bool  ov   = g.overlap_sets;
    const int   ncls = static_cast<int>(uni.semantics().size());
    bool changed = false;

    // global_context bookkeeping: edges deposited this frame (gated after the loop), and a
    // per-frame cache of each root's visible ratio so a big component is scanned at most once.
    std::vector<std::uint64_t>     touched;
    std::unordered_map<int, float> vis_cache;

    for (const ObjectSeed& P : props) {
        // 1. Voxel-dedup the proposal: one representative gidx per occupied voxel.
        std::unordered_map<VKey, int, VHash> pv;
        int maxg = -1;
        for (int gi : P.points) {
            if (gi < 0 || gi >= static_cast<int>(cloud->size())) continue;
            pv.emplace(voxelOf((*cloud)[gi], inv), gi);
            if (gi > maxg) maxg = gi;
        }
        if (pv.empty()) continue;
        ++total_obs_;
        if (static_cast<int>(pt_owner_.size()) <= maxg)
            pt_owner_.resize(static_cast<std::size_t>(maxg) + 1, -1);

        // Class-majority guard: skip an overlap with a component whose dominant class differs.
        auto passClass = [&](int r) -> bool {
            if (!(g.require_class && P.class_id >= 0)) return true;
            int cc = -1, best = -1;
            for (const auto& cv : comp_[r].cls) if (cv.second > best) { best = cv.second; cc = cv.first; }
            return !(cc >= 0 && cc != P.class_id);
        };

        // 2. Tally shared voxels per same-class component root, and collect the voxels this
        //    proposal contributes to its home. Exclusive: `fresh` = the UNOWNED voxels (owned
        //    ones are overlap evidence, never taken); owner located via the single-owner
        //    pt_owner_ map. overlap_sets: EVERY voxel is contributed (non-exclusive), and
        //    overlaps are located via the voxel->{components} index (distinct roots per voxel).
        std::unordered_map<int, int> overlap;                 // component root -> shared voxels
        std::vector<std::pair<VKey, int>> fresh;
        fresh.reserve(pv.size());
        if (!ov) {
            for (const auto& kv : pv) {
                const int gid = kv.second;
                const int ow  = pt_owner_[gid];
                if (ow < 0) { fresh.emplace_back(kv.first, gid); continue; }
                const int r = find(ow);
                if (passClass(r)) ++overlap[r];
            }
        } else {
            std::vector<int> roots;
            for (const auto& kv : pv) {
                fresh.emplace_back(kv.first, kv.second);       // all voxels join the home
                auto it = pt_vox_.find(kv.first);
                if (it == pt_vox_.end()) continue;
                roots.clear();
                for (int leaf : it->second) {                  // distinct roots for this voxel
                    const int r = find(leaf);
                    bool dup = false; for (int x : roots) if (x == r) { dup = true; break; }
                    if (!dup) roots.push_back(r);
                }
                for (int r : roots) if (passClass(r)) ++overlap[r];
            }
        }
        const int np = static_cast<int>(pv.size());

        // Add `vs` to component `root`'s footprint. Exclusive: claim only voxels new to the
        // component and record the single owner in pt_owner_. overlap_sets: add every voxel
        // (a voxel may live in several components) and append `root` to its pt_vox_ owner list
        // (deduped by resolved root). `root` is assumed to be a current Union-Find root.
        auto claimVoxels = [&](int root, const std::vector<std::pair<VKey, int>>& vs) {
            Comp& C = comp_[root];
            for (const auto& f : vs) {
                const bool isnew = C.vox.emplace(f.first, f.second).second;
                if (!ov) { if (isnew) pt_owner_[f.second] = root; }
                else {
                    auto& owners = pt_vox_[f.first];
                    bool present = false;
                    for (int l : owners) if (find(l) == root) { present = true; break; }
                    if (!present) owners.push_back(root);
                }
            }
        };

        // 3. Merge targets: same-class components whose CONTAINMENT (shared / min(|P|,|comp|))
        //    clears that component's level bar (looser as it accumulates support).
        int   primary = -1;
        float best_c  = -1.0f;
        std::vector<int> targets;
        for (const auto& kv : overlap) {
            const int   r     = kv.first;
            const int   csize = static_cast<int>(comp_[r].vox.size());
            const float cont  = static_cast<float>(kv.second)
                              / static_cast<float>(std::max(1, std::min(np, csize)));
            if (cont >= g.thresholdFor(comp_[r].support)) {
                targets.push_back(r);
                if (cont > best_c) { best_c = cont; primary = r; }
            }
        }

        // 4a. LEGACY route (global_context off): union every cleared target immediately -- a
        //     single frame's overlap can bridge two components. Byte-for-byte the old path.
        if (!gc) {
            int root;
            if (primary >= 0) {
                root = primary;
                for (int t : targets) if (t != root) root = unite(root, t);   // progressive bridge
            } else if (!fresh.empty()) {
                root = newComp();
            } else {
                continue;
            }
            claimVoxels(root, fresh);
            Comp& C = comp_[root];
            C.cls[P.class_id] += np;
            if (ov) for (int id : P.inst_ids) if (id >= 0) C.ids.insert(id);
            C.support += 1;
            changed = true;
            continue;
        }

        // 4b. GRAPH route (global_context on): the proposal JOINS its single home component
        //     (support + fresh voxels, as before) but does NOT itself union any other
        //     component. Instead it DEPOSITS confidence-weighted affinity onto the edges from
        //     its home to every other overlapped component -- the actual inter-object merge is
        //     deferred to resolveMerges() once enough multi-view evidence has accumulated.
        int home;
        if      (primary  >= 0)   home = primary;    // best-containment cleared same-class comp
        else if (!fresh.empty())  home = newComp();  // brand-new component from fresh voxels
        else                      home = -1;         // nothing to join; still deposit evidence

        // Deposit node: the home if it has one, else the highest-overlap component (so two
        // components co-occurring in a proposal's footprint still accrue mutual evidence).
        int rep = home;
        if (rep < 0) { int bo = -1; for (const auto& kv : overlap) if (kv.second > bo) { bo = kv.second; rep = kv.first; } }

        if (rep >= 0) {
            // Visible ratio of the proposal this frame (fraction of its voxels in view).
            float visP = 1.0f;
            if (view_vox) {
                int seen = 0;
                for (const auto& kv : pv) if (view_vox->count(kv.first)) ++seen;
                visP = np > 0 ? static_cast<float>(seen) / static_cast<float>(np) : 0.0f;
            }
            for (const auto& kv : overlap) {
                const int C = kv.first;
                if (C == rep) continue;                       // no self-edge

                // Semantic similarity: cosine of the proposal's class histogram against the
                // component's class-vote distribution (the signal the legacy path lacked).
                float sim = 1.0f;
                if (g.use_semantic_affinity && !P.hist.empty()
                    && static_cast<int>(P.hist.size()) == ncls) {
                    std::vector<float> ch(static_cast<std::size_t>(ncls), 0.0f);
                    for (const auto& cv : comp_[C].cls)
                        if (cv.first >= 0 && cv.first < ncls)
                            ch[static_cast<std::size_t>(cv.first)] = static_cast<float>(cv.second);
                    sim = objutil::cosine(P.hist, ch);
                }
                // 2D co-touch bonus: if the proposal and component share a DVIS instance-id
                // group, boost similarity (activates InstanceIdGraph::overlaps).
                if (g.use_instance_ids && idg && !P.inst_ids.empty() && !comp_[C].ids.empty()) {
                    std::vector<int> cids(comp_[C].ids.begin(), comp_[C].ids.end());
                    if (idg->overlaps(P.inst_ids, cids))
                        sim = std::min(1.0f, sim + g.id_bonus);
                }

                // Geometric containment of the proposal in this component this frame
                // (shared voxels / min(|P|,|comp|)), the same signal that gates the legacy
                // path -- folded into the accumulated affinity so a proposal that merely
                // grazes C contributes weak merge evidence even when its class matches. adj
                // stays in [0,1] (sim, cont both in [0,1]) and is comparable to thresholdFor.
                float cont = 1.0f;
                if (g.use_containment) {
                    const int csize = static_cast<int>(comp_[C].vox.size());
                    cont = static_cast<float>(kv.second)
                         / static_cast<float>(std::max(1, std::min(np, csize)));
                }

                // Confidence: product of the two primitives' visible ratios this frame.
                float visC = 1.0f;
                if (view_vox) {
                    auto it = vis_cache.find(C);
                    if (it == vis_cache.end()) { visC = visRatio(comp_[C], view_vox); vis_cache.emplace(C, visC); }
                    else                       { visC = it->second; }
                }
                const float conf = visP * visC;
                if (conf <= 0.0f) continue;

                const std::uint64_t k = ekey(rep, C);
                EdgeAcc& e = edges_[k];
                e.sim_conf += sim * cont * conf;
                e.conf     += conf;
                touched.push_back(k);
            }
        }

        // Join the home component: contribute its voxels + one unit of support + ids.
        if (home >= 0) {
            claimVoxels(home, fresh);
            Comp& C = comp_[home];
            C.cls[P.class_id] += np;
            for (int id : P.inst_ids) if (id >= 0) C.ids.insert(id);
            C.support += 1;
            changed = true;
        }
    }

    if (gc && !touched.empty()) changed = resolveMerges(touched, g) || changed;
    if (changed) { ++version_; rebuildCache(uni, g); }
}

// -- global_context: gated, neighborhood-aggregated inter-component merge -----------------

bool Objects::resolveMerges(const std::vector<std::uint64_t>& touched, const Params& g) {
    // 1. Collapse the deposit-time edge store onto a ROOT-level adjacency snapshot.
    std::unordered_map<std::uint64_t, EdgeAcc> radj;
    for (const auto& e : edges_) {
        int a = find(static_cast<int>(static_cast<std::uint32_t>(e.first >> 32)));
        int b = find(static_cast<int>(static_cast<std::uint32_t>(e.first & 0xffffffffu)));
        if (a == b) continue;                                 // internal edge (already merged)
        EdgeAcc& r = radj[ekey(a, b)];
        r.sim_conf += e.second.sim_conf;
        r.conf     += e.second.conf;
    }
    if (radj.empty()) return false;

    // Adjacency lists for the neighborhood aggregation (judge_connect).
    std::unordered_map<int, std::vector<int>> nbr;
    for (const auto& kv : radj) {
        const int a = static_cast<int>(static_cast<std::uint32_t>(kv.first >> 32));
        const int b = static_cast<int>(static_cast<std::uint32_t>(kv.first & 0xffffffffu));
        nbr[a].push_back(b);
        nbr[b].push_back(a);
    }
    auto adjOf = [&](int a, int b) -> float {
        if (a == b) return 0.0f;
        auto it = radj.find(ekey(a, b));
        if (it == radj.end() || it->second.conf <= 0.0f) return 0.0f;
        return it->second.sim_conf / it->second.conf;
    };
    // Region-aggregated affinity of A to B's whole neighborhood (B weighted full, its
    // neighbors decayed), each weighted by voxel count -- SAI3D judge_connect.
    auto region = [&](int A, int B) -> float {
        double num = 0.0, den = 0.0;
        const double w0 = static_cast<double>(std::max<std::size_t>(1, comp_[B].vox.size()));
        num += w0 * adjOf(A, B); den += w0;
        auto it = nbr.find(B);
        if (it != nbr.end()) for (int K : it->second) {
            if (K == A) continue;
            const double w = g.dis_decay * static_cast<double>(std::max<std::size_t>(1, comp_[K].vox.size()));
            num += w * adjOf(A, K); den += w;
        }
        return den > 0.0 ? static_cast<float>(num / den) : 0.0f;
    };

    // 2. Gate the edges touched this frame against the snapshot; collect surviving merges.
    std::vector<std::pair<int, int>> merges;
    std::unordered_set<std::uint64_t> seen;
    for (std::uint64_t tk : touched) {
        int a = find(static_cast<int>(static_cast<std::uint32_t>(tk >> 32)));
        int b = find(static_cast<int>(static_cast<std::uint32_t>(tk & 0xffffffffu)));
        if (a == b) continue;
        const std::uint64_t rk = ekey(a, b);
        if (!seen.insert(rk).second) continue;                // each root-pair once
        auto it = radj.find(rk);
        if (it == radj.end()) continue;
        const EdgeAcc& e = it->second;
        if (e.conf < g.min_evidence) continue;                // too little multi-view support
        const float adj = e.conf > 0.0f ? e.sim_conf / e.conf : 0.0f;
        const float bar = g.thresholdFor(std::min(comp_[a].support, comp_[b].support));
        if (adj < bar) continue;                              // direct multi-view bar
        if (region(a, b) < bar || region(b, a) < bar) continue;  // neighborhood must agree
        merges.emplace_back(a, b);
    }

    // 3. Apply the surviving merges (computed from one snapshot; unions are associative).
    bool merged = false;
    for (const auto& m : merges)
        if (find(m.first) != find(m.second)) { unite(m.first, m.second); merged = true; }

    // 4. Recompact the edge store onto current roots (drop internal edges, coalesce dups).
    if (merged) {
        std::unordered_map<std::uint64_t, EdgeAcc> compact;
        for (const auto& e : edges_) {
            const int a = find(static_cast<int>(static_cast<std::uint32_t>(e.first >> 32)));
            const int b = find(static_cast<int>(static_cast<std::uint32_t>(e.first & 0xffffffffu)));
            if (a == b) continue;
            EdgeAcc& r = compact[ekey(a, b)];
            r.sim_conf += e.second.sim_conf;
            r.conf     += e.second.conf;
        }
        edges_.swap(compact);
    }
    return merged;
}

// -- global_context: small-region cleanup (SAI3D merge_small_segs) ------------------------

void Objects::mergeSmallComps(const Universe& uni, const Params& g) {
    if (!g.global_context || g.small_seg_min <= 0) return;

    bool merged   = false;
    bool progress = true;
    while (progress) {
        progress = false;
        // Rebuild the root adjacency snapshot each sweep (roots change as we absorb).
        std::unordered_map<std::uint64_t, EdgeAcc> radj;
        for (const auto& e : edges_) {
            const int a = find(static_cast<int>(static_cast<std::uint32_t>(e.first >> 32)));
            const int b = find(static_cast<int>(static_cast<std::uint32_t>(e.first & 0xffffffffu)));
            if (a == b) continue;
            EdgeAcc& r = radj[ekey(a, b)];
            r.sim_conf += e.second.sim_conf;
            r.conf     += e.second.conf;
        }
        std::unordered_map<int, std::pair<int, float>> best;  // root -> (best neighbor, adj)
        for (const auto& kv : radj) {
            const int a = static_cast<int>(static_cast<std::uint32_t>(kv.first >> 32));
            const int b = static_cast<int>(static_cast<std::uint32_t>(kv.first & 0xffffffffu));
            const float v = kv.second.conf > 0.0f ? kv.second.sim_conf / kv.second.conf : 0.0f;
            auto ba = best.find(a); if (ba == best.end() || v > ba->second.second) best[a] = {b, v};
            auto bb = best.find(b); if (bb == best.end() || v > bb->second.second) best[b] = {a, v};
        }
        for (int id = 0; id < static_cast<int>(comp_.size()); ++id) {
            if (parent_[id] != id) continue;                  // not a root
            if (comp_[id].support >= g.small_seg_min) continue;
            auto it = best.find(id);
            if (it == best.end()) continue;                   // isolated -> leave as-is
            const int tgt = find(it->second.first);
            if (tgt == id) continue;
            unite(id, tgt);
            merged = true; progress = true;
        }
    }
    if (merged) { ++version_; rebuildCache(uni, g); }
}
