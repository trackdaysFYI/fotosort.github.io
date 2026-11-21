/* FotoSort Studio ordering. © 2025–2026 FotoSort.com */
const StudioOrder = (() => {
  const CLUSTER_ALPHA = 1.3;
  const DENSITY_K = 7;
  const GAP_WINDOW = 10;

  function makeOrdering(D, n) {
    const dist = (a, b) => D[a * n + b];

    function kthNeighbour(ids, k) {
      const out = new Float64Array(ids.length);
      for (let i = 0; i < ids.length; i += 1) {
        const best = [];
        const row = ids[i] * n;
        for (let j = 0; j < ids.length; j += 1) {
          if (j === i) continue;
          const d = D[row + ids[j]];
          if (best.length < k) { best.push(d); best.sort((a, b) => a - b); }
          else if (d < best[k - 1]) { best[k - 1] = d; best.sort((a, b) => a - b); }
        }
        out[i] = best.length ? best[best.length - 1] : 0;
      }
      return out;
    }

    const all = Array.from({length: n}, (_, i) => i);
    const kth = Array.from(kthNeighbour(all, DENSITY_K)).sort((a, b) => a - b);
    const dc = kth[Math.floor(kth.length / 2)] || 0.3;

    function density(ids) {
      const rho = new Float64Array(ids.length);
      for (let i = 0; i < ids.length; i += 1) {
        const row = ids[i] * n;
        let s = 0;
        for (let j = 0; j < ids.length; j += 1) {
          if (j === i) continue;
          const x = D[row + ids[j]] / dc;
          s += Math.exp(-x * x);
        }
        rho[i] = s;
      }
      return rho;
    }

    function grow(ids, start = 0) {
      const m = ids.length;
      const out = [];
      if (!m) return out;
      const done = new Uint8Array(m);
      const best = new Float64Array(m).fill(Infinity);
      let cur = start;
      while (cur >= 0) {
        done[cur] = 1;
        out.push(ids[cur]);
        const row = ids[cur] * n;
        let next = -1;
        let nextD = Infinity;
        for (let j = 0; j < m; j += 1) {
          if (done[j]) continue;
          const d = D[row + ids[j]];
          if (d < best[j]) best[j] = d;
          if (best[j] < nextD) { nextD = best[j]; next = j; }
        }
        cur = next;
      }
      return out;
    }

    function chain(ids) {
      if (ids.length < 3) return ids.slice();
      const rho = density(ids);
      let start = 0;
      for (let i = 1; i < ids.length; i += 1) if (rho[i] > rho[start]) start = i;
      return grow(ids, start);
    }

    function clusters(ids) {
      const m = ids.length;
      if (m < 3) return ids.slice();
      const rho = density(ids);
      const order = Array.from({length: m}, (_, i) => i).sort((a, b) => rho[b] - rho[a]);
      const label = new Int32Array(m);
      const groups = [];
      for (let r = 0; r < m; r += 1) {
        const i = order[r];
        const row = ids[i] * n;
        let parent = -1;
        let delta = Infinity;
        for (let q = 0; q < r; q += 1) {
          const d = D[row + ids[order[q]]];
          if (d < delta) { delta = d; parent = order[q]; }
        }
        if (parent < 0 || delta > CLUSTER_ALPHA * dc) {
          label[i] = groups.length;
          groups.push({peakRho: rho[i], members: []});
        } else {
          label[i] = label[parent];
        }
        groups[label[i]].members.push(ids[i]);
      }
      groups.sort((a, b) => (b.members.length - a.members.length) || (b.peakRho - a.peakRho));
      return groups.flatMap(g => grow(g.members, 0));
    }

    function gaps(order) {
      const out = new Float64Array(order.length);
      for (let i = 1; i < order.length; i += 1) {
        let best = Infinity;
        for (let j = Math.max(0, i - GAP_WINDOW); j < i; j += 1) best = Math.min(best, dist(order[i], order[j]));
        out[i] = best;
      }
      return out;
    }

    function examples(list) {
      const c = list.length;
      if (c <= 3) return list.slice();
      const total = list.map(a => { let s = 0; for (const b of list) s += D[a * n + b]; return s; });
      const central = list.map((id, i) => [id, total[i]]).sort((a, b) => a[1] - b[1]).map(x => x[0]);
      const out = central.slice(0, 3);
      const chosen = new Set(out);
      const minD = new Map(list.map(id => [id, Math.min(...out.map(o => D[id * n + o]))]));
      while (out.length < c) {
        let pick = -1;
        let far = -1;
        for (const id of list) if (!chosen.has(id) && minD.get(id) > far) { far = minD.get(id); pick = id; }
        out.push(pick);
        chosen.add(pick);
        const row = pick * n;
        for (const id of list) if (!chosen.has(id) && D[row + id] < minD.get(id)) minD.set(id, D[row + id]);
      }
      return out;
    }

    return {dc, chain, clusters, gaps, examples};
  }

  function sample(ids, k) {
    if (ids.length <= k) return ids.slice();
    return Array.from({length: k}, (_, i) => ids[Math.round(i * (ids.length - 1) / (k - 1))]);
  }
  return {makeOrdering, sample};
})();
