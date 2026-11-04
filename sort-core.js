/* FotoSort Studio sorting engine. © 2025–2026 FotoSort.com */
const FS = (() => {
  "use strict";

  const POOL = -1;
  const TRASH = -2;
  const UNASSIGNED = "Unassigned";
  const SETTINGS_KEY = "fotosort-sort-settings";
  function sanitizeName(value) {
    return value.replace(/[^a-zA-Z0-9_ -]/g, "").slice(0, 32);
  }

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value == null || value === false) continue;
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key === "html") node.innerHTML = value;
        else if (key === "style") node.style.cssText = value;
        else if (typeof value === "function" && key.startsWith("on")) node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value === true ? "" : value);
      }
    }
    node.append(...nodes(children));
    return node;
  }

  function nodes(...children) {
    return children.flat(Infinity).filter(child => child != null && child !== false);
  }

  function plural(count, word) {
    return `${count} ${word}${count === 1 ? "" : "s"}`;
  }

  function photoWeight(ds, id) {
    const count = ds.photos[id] && ds.photos[id].count;
    return count > 0 ? count : 1;
  }

  function photosWeight(ds, ids) {
    let total = 0;
    for (const id of ids) total += photoWeight(ds, id);
    return total;
  }

  function readSettings() {
    try { return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {}; }
    catch (err) { return {}; }
  }

  function writeSettings(patch) {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify({...readSettings(), ...patch})); }
    catch (err) { /* storage full or disabled */ }
  }

  /* ---------- data loading (v0 walk, or v1 section lengths) ---------- */

  function parseDataBINV1(arrayBuffer, dataView, decoder) {
    if (arrayBuffer.byteLength < 29) throw new Error("data.bin is shorter than the v1 header");
    const numSources = dataView.getUint16(1, true);
    const numGroups = dataView.getUint16(3, true);
    const minVal = dataView.getFloat32(5, true);
    const maxVal = dataView.getFloat32(9, true);
    const sourcesLen = dataView.getUint32(13, true);
    const groupsLen = dataView.getUint32(17, true);
    const distancesLen = dataView.getUint32(21, true);
    const thumbnailsLen = dataView.getUint32(25, true);
    if (arrayBuffer.byteLength !== 29 + sourcesLen + groupsLen + distancesLen + thumbnailsLen) {
      throw new Error("data.bin length does not match section lengths");
    }
    if (distancesLen !== numGroups * (numGroups - 1)) {
      throw new Error("distance section length does not match group count");
    }

    let cursor = 29;
    const sourcesEnd = cursor + sourcesLen;
    const idToSourceName = {};
    const skippedSourceNames = [];
    for (let i = 0; i < numSources; i += 1) {
      if (cursor + 1 > sourcesEnd) throw new Error("sources section does not match its length");
      const nameLength = dataView.getUint8(cursor); cursor += 1;
      if (cursor + nameLength + 1 > sourcesEnd) throw new Error("sources section does not match its length");
      const name = decoder.decode(new Uint8Array(arrayBuffer, cursor, nameLength)); cursor += nameLength;
      const numNameIds = dataView.getUint8(cursor); cursor += 1;
      if (cursor + numNameIds * 2 > sourcesEnd) throw new Error("sources section does not match its length");
      if (!numNameIds) skippedSourceNames.push(name);
      for (let j = 0; j < numNameIds; j += 1) {
        idToSourceName[dataView.getUint16(cursor, true)] = name; cursor += 2;
      }
    }
    if (cursor !== sourcesEnd) throw new Error("sources section does not match its length");

    const groupsEnd = cursor + groupsLen;
    const imageInfoArr = Array.from({length: numGroups}, () => ({}));
    for (let i = 0; i < numGroups; i += 1) {
      if (cursor + 2 > groupsEnd) throw new Error("groups section does not match its length");
      const numIds = dataView.getUint16(cursor, true); cursor += 2;
      if (numIds < 1) throw new Error("group must contain at least one crop");
      if (cursor + numIds * 2 > groupsEnd) throw new Error("groups section does not match its length");
      const ids = [];
      for (let j = 0; j < numIds; j += 1) {
        ids.push(dataView.getUint16(cursor, true)); cursor += 2;
      }
      imageInfoArr[i].extIds = ids;
    }
    if (cursor !== groupsEnd) throw new Error("groups section does not match its length");

    imageInfoArr.forEach(info => {
      info.srcNames = info.extIds.map(id => idToSourceName[id]).filter(Boolean);
    });

    const n = numGroups;
    const D = new Float32Array(n * n);
    const numPairs = distancesLen / 2;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const d = dataView.getUint16(cursor, true) / 65535 * (maxVal - minVal) + minVal;
        cursor += 2;
        D[i * n + j] = d;
        D[j * n + i] = d;
      }
    }
    if (numPairs !== n * (n - 1) / 2) throw new Error("distance section length does not match group count");

    const thumbnailsEnd = cursor + thumbnailsLen;
    if (cursor + numGroups * 4 > thumbnailsEnd) throw new Error("thumbnail section does not match its length");
    const imageLengths = [];
    for (let i = 0; i < numGroups; i += 1) {
      imageLengths.push(dataView.getUint32(cursor, true)); cursor += 4;
    }
    let thumbnailBytes = 0;
    for (const imageLength of imageLengths) thumbnailBytes += imageLength;
    if (numGroups * 4 + thumbnailBytes !== thumbnailsLen) throw new Error("thumbnail section does not match its length");
    for (let i = 0; i < numGroups; i += 1) {
      const imageLength = imageLengths[i];
      if (cursor + imageLength > thumbnailsEnd) throw new Error("thumbnail section does not match its length");
      imageInfoArr[i].imgData = new Uint8Array(arrayBuffer, cursor, imageLength); cursor += imageLength;
    }
    if (cursor !== thumbnailsEnd) throw new Error("thumbnail section does not match its length");

    return {imageInfoArr, D, skippedSourceNames};
  }

  function parseDataBIN(arrayBuffer) {
    const dataView = new DataView(arrayBuffer);
    const decoder = new TextDecoder("utf-8");
    let cursor = 0;
    const version = dataView.getUint8(cursor); cursor += 1;
    if (version === 1) return parseDataBINV1(arrayBuffer, dataView, decoder);
    if (version !== 0) throw new Error("unsupported data.bin format version: " + version);
    const numImages = dataView.getUint16(cursor, true); cursor += 2;
    const imageInfoArr = Array.from({length: numImages}, () => ({}));

    for (let i = 0; i < numImages; i += 1) {
      const numIds = dataView.getUint16(cursor, true); cursor += 2;
      const ids = [];
      for (let j = 0; j < numIds; j += 1) {
        ids.push(dataView.getUint16(cursor, true)); cursor += 2;
      }
      imageInfoArr[i].extIds = ids;
    }

    const idToSourceName = {};
    const skippedSourceNames = [];
    const numNames = dataView.getUint16(cursor, true); cursor += 2;
    for (let i = 0; i < numNames; i += 1) {
      const nameLength = dataView.getUint8(cursor); cursor += 1;
      const name = decoder.decode(new Uint8Array(arrayBuffer, cursor, nameLength)); cursor += nameLength;
      const numNameIds = dataView.getUint8(cursor); cursor += 1;
      if (!numNameIds) skippedSourceNames.push(name);
      for (let j = 0; j < numNameIds; j += 1) {
        idToSourceName[dataView.getUint16(cursor, true)] = name; cursor += 2;
      }
    }

    imageInfoArr.forEach(info => {
      info.srcNames = info.extIds.map(id => idToSourceName[id]).filter(Boolean);
    });

    for (let i = 0; i < numImages; i += 1) {
      const imageLength = dataView.getUint32(cursor, true); cursor += 4;
      imageInfoArr[i].imgData = new Uint8Array(arrayBuffer, cursor, imageLength); cursor += imageLength;
    }

    const minVal = dataView.getFloat32(cursor, true); cursor += 4;
    const maxVal = dataView.getFloat32(cursor, true); cursor += 4;
    const numPairs = (arrayBuffer.byteLength - cursor) / 2;
    if (numPairs !== numImages * (numImages - 1) / 2) throw new Error("Similarity data does not match photo count");
    const n = numImages;
    const D = new Float32Array(n * n);
    let p = cursor;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const d = dataView.getUint16(p, true) / 65535 * (maxVal - minVal) + minVal;
        p += 2;
        D[i * n + j] = d;
        D[j * n + i] = d;
      }
    }
    return {imageInfoArr, D, skippedSourceNames};
  }

  const API_BASE = "https://9jy8ljy0k0.execute-api.us-east-1.amazonaws.com";

  async function fetchBuffer(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
    return response.arrayBuffer();
  }

  async function fetchSession(sessionId, name, includeBbox) {
    if (typeof getBearer !== "function") throw new Error("Auth is not available on this page");
    const params = new URLSearchParams({sessionId});
    if (name) params.set("name", name);
    if (includeBbox) params.set("bbox", "True");
    const authorization = await getBearer();
    const response = await fetch(`${API_BASE}/getSession?${params}`, {
      method: "GET",
      headers: {"Authorization": authorization}
    });
    if (!response.ok) throw new Error(`getSession HTTP ${response.status}`);
    return response.json();
  }

  function sourceNameFor(origUrls, names) {
    if (!origUrls || !names) return null;
    for (const name of names) {
      if (name && origUrls[name]) return name;
    }
    return null;
  }

  function originalUrlForPhoto(origUrls, names) {
    const name = sourceNameFor(origUrls, names);
    return name ? origUrls[name] : null;
  }

  function imageSize(bytes) {
    const n = bytes ? bytes.length : 0;
    if (n >= 30 && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 &&
        bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50) {
      const tag = String.fromCharCode(bytes[12], bytes[13], bytes[14], bytes[15]);
      if (tag === "VP8 " && bytes[23] === 0x9d && bytes[24] === 0x01 && bytes[25] === 0x2a) {
        const w = (bytes[26] | (bytes[27] << 8)) & 0x3fff;
        const h = (bytes[28] | (bytes[29] << 8)) & 0x3fff;
        if (w && h) return [w, h];
      } else if (tag === "VP8L" && bytes[20] === 0x2f) {
        const b0 = bytes[21], b1 = bytes[22], b2 = bytes[23], b3 = bytes[24];
        const w = 1 + (((b1 & 0x3f) << 8) | b0);
        const h = 1 + (((b3 & 0x0f) << 10) | (b2 << 2) | ((b1 & 0xc0) >> 6));
        if (w && h) return [w, h];
      } else if (tag === "VP8X") {
        const w = 1 + (bytes[24] | (bytes[25] << 8) | (bytes[26] << 16));
        const h = 1 + (bytes[27] | (bytes[28] << 8) | (bytes[29] << 16));
        if (w && h) return [w, h];
      }
    }
    if (n > 9 && bytes[0] === 0xff && bytes[1] === 0xd8) {
      let i = 2;
      while (i + 9 < n) {
        if (bytes[i] !== 0xff) break;
        const marker = bytes[i + 1];
        if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) { i += 2; continue; }
        const len = (bytes[i + 2] << 8) | bytes[i + 3];
        if (len < 2 || i + 2 + len > n) break;
        if (marker >= 0xc0 && marker <= 0xc3) {
          const h = (bytes[i + 5] << 8) | bytes[i + 6];
          const w = (bytes[i + 7] << 8) | bytes[i + 8];
          if (w && h) return [w, h];
        }
        i += 2 + len;
      }
    }
    return null;
  }

  function medianRatio(values) {
    if (!values.length) return 1;
    const ratios = values.slice().sort((a, b) => a - b);
    const mid = ratios.length >> 1;
    const aspect = ratios.length % 2 ? ratios[mid] : (ratios[mid - 1] + ratios[mid]) / 2;
    return Math.round(aspect * 10000) / 10000;
  }

  function labelInk(hex) {
    const raw = String(hex || "").replace("#", "");
    if (raw.length !== 6) return "#111";
    const n = parseInt(raw, 16);
    if (!Number.isFinite(n)) return "#111";
    const r = (n >> 16) & 255;
    const g = (n >> 8) & 255;
    const b = n & 255;
    return (r * 299 + g * 587 + b * 114) > 150000 ? "#111" : "#fff";
  }

  function fitLightboxFrame(frame, img) {
    const maxW = Math.min(window.innerWidth * 0.9, 1200);
    const maxH = window.innerHeight * 0.78;
    const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
    frame.style.width = Math.max(1, Math.round(img.naturalWidth * scale)) + "px";
    frame.style.height = Math.max(1, Math.round(img.naturalHeight * scale)) + "px";
  }

  async function loadDataset(cfg = {}) {
    const params = new URLSearchParams(location.search);
    const sessionId = params.get("sessionId");
    const sample = params.get("data") || "data.bin";

    let buffer = null;
    let fromSession = false;
    let notice = "";
    let sessionName = "";
    let bboxes = {};

    if (sessionId) {
      try {
        const access = await fetchSession(sessionId, null, cfg.bbox);
        sessionName = access.name || "";
        if (access.bbox) bboxes = access.bbox;
        buffer = await fetchBuffer(access.dataUrl);
        fromSession = true;
      } catch (err) {
        console.error(err);
        notice = `Could not open session "${sessionId.slice(0, 8)}". Showing the sample data.bin instead.`;
      }
    }
    if (!buffer) buffer = await fetchBuffer(sample);

    const parsed = parseDataBIN(buffer);
    const photos = parsed.imageInfoArr.map((info, id) => {
      const size = imageSize(info.imgData);
      const aspect = size ? Math.round((size[0] / size[1]) * 10000) / 10000 : 1;
      return {
        id,
        url: URL.createObjectURL(new Blob([info.imgData], {type: "image/webp"})),
        names: info.srcNames,
        extIds: info.extIds,
        count: info.extIds.length,
        label: info.srcNames[0] || `Crop ${id + 1}`,
        aspect
      };
    });
    const aspect = medianRatio(photos.map(photo => photo.aspect).filter(value => value > 0));
    return {
      n: photos.length,
      D: parsed.D,
      photos,
      aspect,
      skipped: parsed.skippedSourceNames,
      sessionId,
      sessionName,
      bboxes,
      bboxesByPhoto: {},
      origUrls: {},
      fromSession,
      sourceLabel: fromSession ? (sessionName || "Session") : sample,
      notice
    };
  }

  /* ---------- undoable store ---------- */

  class Store {
    constructor(n) {
      this.n = n;
      this.loc = new Int32Array(n).fill(POOL);
      this.stamp = new Float64Array(n);
      this.groups = [];
      this.seq = 0;
      this.tick = 0;
      this.version = 0;
      this.past = [];
      this.future = [];
      this.onChange = null;
    }

    snapshot() {
      return {
        loc: this.loc.slice(),
        stamp: this.stamp.slice(),
        groups: this.groups.map(g => ({...g})),
        seq: this.seq,
        tick: this.tick
      };
    }

    restore(state) {
      this.loc = state.loc.slice();
      this.stamp = state.stamp.slice();
      this.groups = state.groups.map(g => ({...g}));
      this.seq = state.seq;
      this.tick = state.tick;
    }

    emit() {
      this.version += 1;
      if (this.onChange) this.onChange("data");
    }

    commit(label, fn) {
      const before = this.snapshot();
      const result = fn();
      this.past.push({label, state: before});
      if (this.past.length > 300) this.past.shift();
      this.future = [];
      this.emit();
      return result;
    }

    undo() {
      const entry = this.past.pop();
      if (!entry) return null;
      this.future.push({label: entry.label, state: this.snapshot()});
      this.restore(entry.state);
      this.emit();
      return entry.label;
    }

    redo() {
      const entry = this.future.pop();
      if (!entry) return null;
      this.past.push({label: entry.label, state: this.snapshot()});
      this.restore(entry.state);
      this.emit();
      return entry.label;
    }

    group(id) {
      return this.groups.find(g => g.id === id) || null;
    }

    place(ids, dest) {
      for (const id of ids) {
        this.tick += 1;
        this.loc[id] = dest;
        this.stamp[id] = this.tick;
      }
    }

    addGroup(name) {
      this.seq += 1;
      const group = {
        id: this.seq,
        name: name || `Group ${this.seq}`
      };
      this.groups.push(group);
      return group;
    }

    removeGroup(id) {
      for (let i = 0; i < this.n; i += 1) if (this.loc[i] === id) this.loc[i] = POOL;
      this.groups = this.groups.filter(g => g.id !== id);
    }
  }

  /* ---------- app ---------- */

  class App {
    constructor(ds, cfg) {
      this.ds = ds;
      this.cfg = cfg;
      this.store = new Store(ds.n);
      this.store.onChange = kind => this.update(kind);
      this.sel = new Set();
      this.anchor = null;
      const saved = readSettings();
      this.settings = {
        unsorted: saved.unsorted !== false
      };
      this.zoom = saved.zoom || saved["zoom:studio.html"] || cfg.zoom || 96;
      this.tiles = new Map();
      this.render = null;
      this.state = null;
      this.drag = null;
      this.lightbox = null;
      this.overlay = null;
      this.lastClick = null;
      this._saveTimer = null;
      this.progressKey = `fotosort-progress:${ds.sessionId || "sample"}:${fingerprint(ds)}`;
      document.documentElement.style.setProperty("--tile", this.zoom + "px");
      document.documentElement.style.setProperty("--photo-ar", String(ds.aspect > 0 ? ds.aspect : 1));
    }

    /* ----- derived state ----- */

    compute() {
      const {ds, store} = this;
      const n = ds.n;
      const members = new Map(store.groups.map(g => [g.id, []]));
      const trash = [];
      const pool = [];
      for (let i = 0; i < n; i += 1) {
        const loc = store.loc[i];
        if (loc === POOL) pool.push(i);
        else if (loc === TRASH) trash.push(i);
        else if (members.has(loc)) members.get(loc).push(i);
        else { store.loc[i] = POOL; pool.push(i); }
      }
      trash.sort((a, b) => store.stamp[b] - store.stamp[a]);

      let poolCount = 0;
      let trashCount = 0;
      let sortedCount = 0;
      for (let i = 0; i < n; i += 1) {
        const weight = photoWeight(ds, i);
        const loc = store.loc[i];
        if (loc === POOL) poolCount += weight;
        else if (loc === TRASH) trashCount += weight;
        else sortedCount += weight;
      }
      this.state = {
        pool,
        members,
        trash,
        counts: {
          total: poolCount + trashCount + sortedCount,
          pool: poolCount,
          trash: trashCount,
          sorted: sortedCount,
          groups: store.groups.length
        }
      };
      return this.state;
    }

    photoCountAll(ids) {
      return photosWeight(this.ds, ids);
    }

    rankWith(ref, members) {
      const {ds, store} = this;
      const n = ds.n;
      const D = ds.D;
      const rows = store.groups.map(group => {
        const list = members.get(group.id) || [];
        let dist = Infinity;
        if (list.length && ref.length) {
          let total = 0;
          let counted = 0;
          for (const r of ref) {
            let best = Infinity;
            const rr = r * n;
            for (const m of list) {
              if (m === r) continue;
              const d = D[rr + m];
              if (d < best) best = d;
            }
            if (best === Infinity) continue;
            total += best;
            counted += 1;
          }
          if (counted) dist = total / counted;
        }
        const sorted = list.slice();
        if (ref.length) {
          const r0 = ref[0] * n;
          sorted.sort((a, b) => D[r0 + a] - D[r0 + b]);
        } else {
          sorted.sort((a, b) => store.stamp[b] - store.stamp[a]);
        }
        return {group, id: group.id, dist, members: sorted};
      });
      return rows.sort((a, b) => {
        const ea = a.members.length === 0;
        const eb = b.members.length === 0;
        if (ea !== eb) return ea ? 1 : -1;
        return (a.dist - b.dist) || (a.id - b.id);
      });
    }

    update(kind = "data") {
      this.compute();
      this.syncSelection();
      this.updateChrome();
      if (this.render) this.render(this.state, kind);
      if (kind === "data") this.scheduleSave();
    }

    /* ----- selection ----- */

    orderedSelection(pool = this.state ? this.state.pool : []) {
      const inPool = pool.filter(id => this.sel.has(id));
      const others = [...this.sel].filter(id => this.store.loc[id] !== POOL).sort((a, b) => a - b);
      return inPool.concat(others);
    }

    setSelection(ids) {
      this.sel = new Set(ids);
      this.anchor = ids.length ? ids[ids.length - 1] : null;
      this.update("selection");
    }

    toggle(id) {
      if (this.sel.has(id)) this.sel.delete(id);
      else this.sel.add(id);
      this.anchor = id;
      this.update("selection");
    }

    clearSelection() {
      if (!this.sel.size) return;
      this.sel.clear();
      this.anchor = null;
      this.update("selection");
    }

    syncSelection() {
      for (const map of this.tiles.values()) {
        for (const [id, node] of map) node.classList.toggle("sel", this.sel.has(id));
      }
    }

    /* ----- mutations ----- */

    drop(dest, ids, opts = {}) {
      ids = [...new Set(ids)].filter(id => Number.isInteger(id) && id >= 0 && id < this.ds.n);
      if (!ids.length) {
        this.toast("Nothing selected");
        return null;
      }
      const destLoc = dest === "trash" ? TRASH : dest === "pool" ? POOL : dest === "new" ? null :
        (typeof dest === "number" ? dest : parseInt(String(dest).replace("g:", ""), 10));
      if (destLoc != null) {
        ids = ids.filter(id => this.store.loc[id] !== destLoc);
        if (!ids.length) return null;
      }
      const noun = plural(photosWeight(this.ds, ids), "photo");
      let result = null;
      let message = "";
      this.sel.clear();
      this.anchor = null;
      if (dest === "new") {
        const name = opts.name || `Group ${this.store.seq + 1}`;
        message = `Created ${name} with ${noun}`;
        result = this.store.commit(message, () => {
          const group = this.store.addGroup(name);
          this.store.place(ids, group.id);
          return group;
        });
      } else if (dest === "trash") {
        message = `Discarded ${noun}`;
        this.store.commit(message, () => this.store.place(ids, TRASH));
      } else if (dest === "pool") {
        message = `Returned ${noun} to Unsorted`;
        this.store.commit(message, () => this.store.place(ids, POOL));
      } else {
        const gid = typeof dest === "number" ? dest : parseInt(String(dest).replace("g:", ""), 10);
        result = this.store.group(gid);
        if (!result) return null;
        message = `Moved ${noun} to ${result.name}`;
        this.store.commit(message, () => this.store.place(ids, gid));
      }
      if (!opts.quiet) this.toast(message, {undo: true});
      return result;
    }

    renameGroup(id, name) {
      const group = this.store.group(id);
      const clean = sanitizeName(name).trim();
      if (!group || !clean || clean === group.name) return;
      this.store.commit(`Renamed ${group.name} to ${clean}`, () => {
        this.store.group(id).name = clean;
      });
    }

    deleteGroup(id) {
      const group = this.store.group(id);
      if (!group) return;
      const count = photosWeight(this.ds, this.state.members.get(id) || []);
      if (count > 1 && !window.confirm(`Delete ${group.name}? Its ${count} photos go back to Unsorted.`)) return;
      const message = `Deleted ${group.name}`;
      this.store.commit(message, () => this.store.removeGroup(id));
      this.toast(message, {undo: true});
    }

    undo() {
      const label = this.store.undo();
      this.toast(label ? `Undid: ${label}` : "Nothing to undo");
    }

    redo() {
      const label = this.store.redo();
      this.toast(label ? `Redid: ${label}` : "Nothing to redo");
    }

    duplicateNames() {
      const counts = new Map();
      for (const g of this.store.groups) {
        if (!(this.state.members.get(g.id) || []).length) continue;
        const key = g.name.trim().toLowerCase();
        counts.set(key, (counts.get(key) || 0) + 1);
      }
      const dup = new Set();
      for (const [key, count] of counts) {
        if (count > 1 || (this.settings.unsorted && key === UNASSIGNED.toLowerCase())) dup.add(key);
      }
      return dup;
    }

    /* ----- persistence ----- */

    packProgress() {
      const trash = [];
      const members = {};
      for (let i = 0; i < this.store.n; i += 1) {
        const loc = this.store.loc[i];
        if (loc === POOL) continue;
        if (loc === TRASH) {
          trash.push(i);
          continue;
        }
        const key = String(loc);
        if (!members[key]) members[key] = [];
        members[key].push(i);
      }
      return {
        v: 1,
        trash,
        members,
        groups: this.store.groups.map(g => ({id: g.id, name: g.name})),
        seq: this.store.seq
      };
    }

    scheduleSave() {
      clearTimeout(this._saveTimer);
      this._saveTimer = setTimeout(() => {
        try { localStorage.setItem(this.progressKey, JSON.stringify(this.packProgress())); }
        catch (err) { /* storage full */ }
      }, 250);
    }

    restoreProgress() {
      let saved = null;
      try { saved = JSON.parse(localStorage.getItem(this.progressKey)); }
      catch (err) { return false; }
      if (!saved || saved.v !== 1 || typeof saved.members !== "object" || Array.isArray(saved.loc)) return false;

      const groups = saved.groups || [];
      const ids = new Set(groups.map(g => g.id));
      this.store.groups = groups.map(g => ({id: g.id, name: g.name}));
      this.store.seq = saved.seq || Math.max(0, ...ids, 0);
      this.store.loc.fill(POOL);
      let moved = false;
      for (const id of saved.trash || []) {
        if (id >= 0 && id < this.ds.n) {
          this.store.loc[id] = TRASH;
          moved = true;
        }
      }
      for (const [gidStr, list] of Object.entries(saved.members || {})) {
        const gid = Number(gidStr);
        if (!ids.has(gid) || !Array.isArray(list)) continue;
        for (const id of list) {
          if (id >= 0 && id < this.ds.n) {
            this.store.loc[id] = gid;
            moved = true;
          }
        }
      }
      for (let i = 0; i < this.ds.n; i += 1) this.store.stamp[i] = i;
      this.store.tick = this.ds.n;
      this.store.version += 1;
      return moved;
    }

    startOver() {
      this.store.commit("Start over", () => {
        this.store.loc.fill(POOL);
        this.store.groups = [];
        this.store.seq = 0;
      });
      this.toast("Cleared all groups", {undo: true});
    }

    /* ----- tiles ----- */

    tile(id, variant = "main") {
      let map = this.tiles.get(variant);
      if (!map) {
        map = new Map();
        this.tiles.set(variant, map);
      }
      let node = map.get(id);
      if (!node) {
        const photo = this.ds.photos[id];
        node = el("div", {class: "ph", "data-id": id, title: photo.names.join("\n") || photo.label},
          el("img", {src: photo.url, alt: "", draggable: "false", decoding: "async"}),
          photo.count > 1 ? el("span", {class: "ph-count", text: photo.count + "x"}) : null);
        if (photo.aspect > 0) node.style.setProperty("--ar", String(photo.aspect));
        node.classList.toggle("sel", this.sel.has(id));
        map.set(id, node);
      }
      return node;
    }

    tilesFor(ids) {
      const out = [];
      for (const map of this.tiles.values()) {
        for (const id of ids) if (map.has(id)) out.push(map.get(id));
      }
      return out;
    }

    /* ----- drag and drop ----- */

    enableDrag(root, opts = {}) {
      const selector = opts.selector || ".ph[data-id]";
      root.addEventListener("pointerdown", event => {
        if (event.button !== 0) return;
        const source = event.target.closest(selector);
        if (!source || !root.contains(source)) return;
        if (event.target.closest("input, button, a, select")) return;
        if (opts.filter && !opts.filter(source, event)) return;
        event.preventDefault();
        const start = {x: event.clientX, y: event.clientY};
        let dragging = false;
        const move = ev => {
          if (!dragging) {
            if (Math.hypot(ev.clientX - start.x, ev.clientY - start.y) < 6) return;
            const ids = opts.getIds ? opts.getIds(source) : this.defaultDragIds(source);
            if (!ids || !ids.length) return;
            dragging = true;
            this.beginDrag(ids, ev, opts);
          }
          this.dragMove(ev);
        };
        const finish = ev => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", finish);
          window.removeEventListener("pointercancel", finish);
          if (dragging) {
            if (ev.type === "pointercancel") this.cancelDrag();
            else this.endDrag(ev);
          } else if (ev.type === "pointerup") {
            this.handleClick(source, ev, opts);
          }
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", finish);
        window.addEventListener("pointercancel", finish);
      });
    }

    defaultDragIds(tile) {
      const id = Number(tile.dataset.id);
      return this.sel.has(id) ? this.orderedSelection() : [id];
    }

    handleClick(source, event, opts) {
      const now = performance.now();
      const id = source.dataset.id != null ? Number(source.dataset.id) : null;
      const last = this.lastClick;
      if (id != null && last && last.id === id && now - last.t < 320) {
        this.lastClick = null;
        if (last.undo) last.undo();
        if (opts.doubleClickPreview !== false) this.preview(id);
        return;
      }
      if (opts.onClick) {
        const handled = opts.onClick(source, event);
        if (handled) {
          this.lastClick = {id, t: now, undo: typeof handled === "function" ? handled : null};
          return;
        }
      }
      if (id == null) return;
      const wasSelected = this.sel.has(id);
      if (event.shiftKey && this.anchor != null) {
        const scope = source.closest("[data-scope]") || source.parentElement;
        const order = [...scope.querySelectorAll(".ph[data-id]")].map(node => Number(node.dataset.id));
        const a = order.indexOf(this.anchor);
        const b = order.indexOf(id);
        if (a >= 0 && b >= 0) {
          const [lo, hi] = a < b ? [a, b] : [b, a];
          for (let i = lo; i <= hi; i += 1) this.sel.add(order[i]);
          this.anchor = id;
          this.update("selection");
          this.lastClick = {id, t: now, undo: null};
          return;
        }
      }
      this.toggle(id);
      this.lastClick = {id, t: now, undo: () => { if (wasSelected) this.sel.add(id); else this.sel.delete(id); this.update("selection"); }};
    }

    beginDrag(ids, event, opts = {}) {
      const lead = this.ds.photos[ids[0]];
      const fit = document.documentElement.dataset.fit || "lock";
      const ar = fit === "lock"
        ? (lead && lead.aspect > 0 ? lead.aspect : 1)
        : (this.ds.aspect > 0 ? this.ds.aspect : 1);
      const ghost = el("div", {class: "fs-ghost", style: `width:${76 * ar}px`},
        ids.slice(0, 3).reverse().map((id, i, arr) => el("img", {
          src: this.ds.photos[id].url,
          alt: "",
          style: `transform: rotate(${(arr.length - 1 - i) * -5}deg) translate(${(arr.length - 1 - i) * -4}px, ${(arr.length - 1 - i) * 3}px)`
        })),
        el("span", {class: "fs-ghost-count", text: String(ids.length)}));
      document.body.append(ghost);
      const lifted = this.tilesFor(ids);
      for (const node of lifted) node.classList.add("lifting");
      document.body.classList.add("fs-dragging");
      this.drag = {ids, opts, ghost, lifted, target: null, x: event.clientX, y: event.clientY, raf: 0};
      if (opts.onStart) opts.onStart(ids);
      const loop = () => {
        if (!this.drag) return;
        this.autoScroll();
        this.drag.raf = requestAnimationFrame(loop);
      };
      this.drag.raf = requestAnimationFrame(loop);
      this.dragMove(event);
    }

    dragMove(event) {
      const drag = this.drag;
      if (!drag) return;
      drag.x = event.clientX;
      drag.y = event.clientY;
      drag.ghost.style.transform = `translate(${drag.x + 14}px, ${drag.y + 10}px)`;
      this.updateDropTarget();
    }

    updateDropTarget() {
      const drag = this.drag;
      const under = document.elementFromPoint(drag.x, drag.y);
      const target = under ? under.closest("[data-drop]") : null;
      if (target !== drag.target) {
        if (drag.target) drag.target.classList.remove("drop-hover");
        if (target) target.classList.add("drop-hover");
        drag.target = target;
      }
      drag.under = under;
    }

    autoScroll() {
      const drag = this.drag;
      const under = drag.under;
      const scroller = under ? under.closest("[data-autoscroll]") : null;
      if (!scroller) return;
      const rect = scroller.getBoundingClientRect();
      const edge = 56;
      let vx = 0;
      let vy = 0;
      if (drag.y < rect.top + edge) vy = -Math.ceil((rect.top + edge - drag.y) / 3);
      else if (drag.y > rect.bottom - edge) vy = Math.ceil((drag.y - rect.bottom + edge) / 3);
      if (drag.x < rect.left + edge) vx = -Math.ceil((rect.left + edge - drag.x) / 3);
      else if (drag.x > rect.right - edge) vx = Math.ceil((drag.x - rect.right + edge) / 3);
      if (vx || vy) {
        scroller.scrollBy(vx, vy);
        this.updateDropTarget();
      }
    }

    cleanupDrag() {
      const drag = this.drag;
      if (!drag) return null;
      cancelAnimationFrame(drag.raf);
      drag.ghost.remove();
      for (const node of drag.lifted) node.classList.remove("lifting");
      if (drag.target) drag.target.classList.remove("drop-hover");
      document.body.classList.remove("fs-dragging");
      this.drag = null;
      if (drag.opts.onEnd) drag.opts.onEnd();
      return drag;
    }

    cancelDrag() {
      this.cleanupDrag();
    }

    endDrag() {
      const drag = this.cleanupDrag();
      if (!drag || !drag.target) return;
      const dest = drag.target.dataset.drop;
      if (drag.opts.onDrop && drag.opts.onDrop(dest, drag.ids, drag.target) === true) return;
      if (dest === "none") return;
      this.drop(dest, drag.ids);
    }

    /* ----- keyboard ----- */

    enableKeys() {
      window.addEventListener("keydown", event => {
        if (this.drag && event.key === "Escape") { event.preventDefault(); this.cancelDrag(); return; }
        if (this.lightbox) {
          const inToggle = event.target && event.target.closest && event.target.closest(".fs-lb-toggle");
          if (event.key === "Escape" || (event.key === " " && !inToggle)) {
            event.preventDefault();
            this.closePreview();
          }
          return;
        }
        if (this.overlay) {
          if (event.key === "Escape") { event.preventDefault(); this.closeOverlay(); }
          return;
        }
        const target = event.target;
        if (target && target.matches && target.matches("input[type=text], input[type=search], input:not([type]), textarea, select, [contenteditable]")) return;
        const mod = event.ctrlKey || event.metaKey;
        const key = event.key;
        const lower = key.toLowerCase();
        if (mod && lower === "z" && !event.shiftKey) { event.preventDefault(); this.undo(); return; }
        if (mod || event.altKey) return;
        if (this.cfg.onKey && this.cfg.onKey(event, this) === true) { event.preventDefault(); return; }
        if (key === "Escape") { this.clearSelection(); return; }
        if (key === "?") { event.preventDefault(); this.showHelp(); return; }
        if ((key === "Delete" || key === "Backspace") && this.sel.size) {
          event.preventDefault();
          this.drop("trash", this.orderedSelection());
        }
      });
    }

    /* ----- chrome ----- */

    mountChrome() {
      const header = document.querySelector("[data-fs-header]");
      if (!header) return;
      const brand = this.cfg.logo
        ? el("a", {class: "fs-brand logo", href: "console.html", html: `<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="#FE621D" aria-hidden="true"><path d="M12 8c-2.168 0-4 1.832-4 4s1.832 4 4 4 4-1.832 4-4-1.832-4-4-4zm0 6c-1.065 0-2-.935-2-2s.935-2 2-2 2 .935 2 2-.935 2-2 2z"/><path d="M20 5h-2.586l-2.707-2.707A.996.996 0 0 0 14 2h-4a.996.996 0 0 0-.707.293L6.586 5H4c-1.103 0-2 .897-2 2v11c0 1.103.897 2 2 2h16c1.103 0 2-.897 2-2V7c0-1.103-.897-2-2-2zM4 18V7h3c.266 0 .52-.105.707-.293L10.414 4h3.172l2.707 2.707A.996.996 0 0 0 17 7h3l.002 11H4z"/></svg><strong>FotoSort</strong>`})
        : el("div", {class: "fs-brand"}, "FotoSort ", this.cfg.title ? el("small", {text: this.cfg.title}) : null);
      this.statsText = el("span", {class: "fs-stats-text"});
      this.statsSorted = el("i", {class: "seg-sorted"});
      this.statsPool = el("i", {class: "seg-pool"});
      this.statsTrash = el("i", {class: "seg-trash"});
      this.statsBar = el("div", {class: "fs-bar"}, this.statsSorted, this.statsPool, this.statsTrash);
      const zoomCfg = this.cfg.zoomRange || [64, 200];
      const zoom = el("input", {type: "range", min: String(zoomCfg[0]), max: String(zoomCfg[1]), step: "4", value: String(this.zoom)});
      zoom.addEventListener("input", () => {
        this.zoom = Number(zoom.value);
        document.documentElement.style.setProperty("--tile", this.zoom + "px");
        writeSettings({zoom: this.zoom});
        if (this.cfg.onZoom) this.cfg.onZoom(this.zoom);
      });
      this.undoBtn = el("button", {class: "btn", type: "button", title: "Undo (Ctrl+Z)", onclick: () => this.undo(), text: "Undo"});
      this.redoBtn = el("button", {class: "btn", type: "button", title: "Redo", onclick: () => this.redo(), text: "Redo"});
      const help = el("button", {class: "btn icon", type: "button", title: "Help", onclick: () => this.showHelp(), text: "?"});
      const exportBtn = el("button", {class: "btn primary", type: "button", text: "Export script"});
      exportBtn.addEventListener("click", () => this.showExport(exportBtn));
      const extra = el("div", {class: "fs-extra"});
      header.classList.add("fs-top");
      header.replaceChildren(...nodes(
        brand,
        el("div", {class: "fs-stats"}, this.statsText, this.statsBar),
        el("div", {class: "fs-spacer"}),
        extra,
        this.cfg.zoom === false ? null : el("label", {class: "fs-field"}, "Size", zoom),
        this.undoBtn,
        this.redoBtn,
        help,
        exportBtn
      ));
      this.extraSlot = extra;
      if (this.ds.notice) {
        header.after(el("div", {class: "fs-notice", text: this.ds.notice}));
      }
    }

    updateChrome() {
      if (!this.statsText) return;
      const c = this.state.counts;
      const total = Math.max(1, c.total);
      let donePct = Math.round(100 * (c.sorted + c.trash) / total);
      if (c.pool > 0 && donePct >= 100) donePct = 99;
      this.statsText.textContent = `${donePct}% · ${c.sorted} of ${c.total} sorted · ${c.trash} discarded`;
      this.statsSorted.style.width = (100 * c.sorted / total) + "%";
      this.statsPool.style.width = (100 * c.pool / total) + "%";
      this.statsTrash.style.width = (100 * c.trash / total) + "%";
      this.undoBtn.disabled = !this.store.past.length;
      this.redoBtn.disabled = !this.store.future.length;
    }

    /* ----- overlays ----- */

    toast(message, opts = {}) {
      let host = document.querySelector(".fs-toasts");
      if (!host) {
        host = el("div", {class: "fs-toasts"});
        document.body.append(host);
      }
      const node = el("div", {class: "fs-toast"}, el("span", {text: message}),
        opts.undo ? el("button", {type: "button", text: "Undo", onclick: () => { node.remove(); this.undo(); }}) : null,
        opts.action ? el("button", {type: "button", text: opts.action.label, onclick: () => { node.remove(); opts.action.fn(); }}) : null);
      host.append(node);
      while (host.children.length > 3) host.firstElementChild.remove();
      setTimeout(() => node.classList.add("out"), opts.action ? 9000 : 4200);
      setTimeout(() => node.remove(), opts.action ? 9400 : 4600);
    }

    async ensureOriginalUrls(names) {
      if (!this.ds.fromSession || !this.ds.sessionId || !names || !names.length) return null;
      const cached = originalUrlForPhoto(this.ds.origUrls, names);
      if (cached) return cached;
      const name = names.find(Boolean);
      if (!name) return null;
      try {
        const access = await fetchSession(this.ds.sessionId, name, this.cfg.bbox);
        if (access.imageUrl) this.ds.origUrls[name] = access.imageUrl;
        if (access.bbox) {
          Object.assign(this.ds.bboxes, access.bbox);
          this.ds.bboxesByPhoto[name] = access.bbox;
        }
      } catch (err) {
        console.error(err);
      }
      return originalUrlForPhoto(this.ds.origUrls, names);
    }

    cropOwnerIndex() {
      if (this.ds.cropOwners) return this.ds.cropOwners;
      const map = new Map();
      this.ds.photos.forEach((photo, index) => {
        for (const cropId of photo.extIds || []) map.set(String(cropId), index);
      });
      this.ds.cropOwners = map;
      return map;
    }

    lightboxBoxFlags() {
      const saved = readSettings();
      return {
        current: saved.lightboxCurrentBox !== false,
        others: saved.lightboxOtherBoxes === true
      };
    }

    lightboxLabel(photoIndex) {
      const loc = this.store.loc[photoIndex];
      if (loc === TRASH) return "Discarded";
      const group = loc != null && loc >= 0 ? this.store.group(loc) : null;
      return group ? group.name : "Unsorted";
    }

    paintLightboxBoxes(frame, sourceName, focusExtIds, show) {
      frame.querySelectorAll(".fs-lb-box").forEach(node => node.remove());
      const boxes = this.ds.bboxesByPhoto && this.ds.bboxesByPhoto[sourceName];
      if (!boxes) return;
      const focus = new Set((focusExtIds || []).map(cropId => String(cropId)));
      const owners = this.cropOwnerIndex();
      const layers = {others: [], current: []};
      for (const [cropId, coords] of Object.entries(boxes)) {
        if (!Array.isArray(coords) || coords.length < 4) continue;
        const mine = focus.has(String(cropId));
        if (mine ? !show.current : !show.others) continue;
        const x1 = Number(coords[0]);
        const y1 = Number(coords[1]);
        const x2 = Number(coords[2]);
        const y2 = Number(coords[3]);
        if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
        const owner = owners.get(String(cropId));
        const color = mine ? "#22c55e" : "#ef4444";
        const name = owner == null ? "" : this.lightboxLabel(owner);
        layers[mine ? "current" : "others"].push(el("div", {
          class: "fs-lb-box" + (mine ? " mine" : ""),
          style: `left:${x1 * 100}%;top:${y1 * 100}%;width:${Math.max(0, x2 - x1) * 100}%;height:${Math.max(0, y2 - y1) * 100}%;--bc:${color};--ink:${labelInk(color)}`
        }, name ? el("span", {class: "fs-lb-tag", text: name}) : null));
      }
      frame.append(...layers.others, ...layers.current);
    }

    preview(id) {
      this.closePreview();
      const photo = this.ds.photos[id];
      if (!photo) return;
      const frame = el("div", {class: "fs-lb-frame"});
      const img = el("img", {class: "fs-lb-img", src: photo.url, alt: ""});
      const spinner = el("div", {
        class: "fs-lb-spinner",
        role: "status",
        "aria-live": "polite",
        "aria-label": "Loading full image"
      });
      frame.append(img, spinner);
      const flags = this.lightboxBoxFlags();
      const currentInput = el("input", {type: "checkbox"});
      currentInput.checked = flags.current;
      const othersInput = el("input", {type: "checkbox"});
      othersInput.checked = flags.others;
      const currentToggle = el("label", {class: "fs-lb-toggle current", title: "Show the box for this crop"}, currentInput, "Current");
      const othersToggle = el("label", {class: "fs-lb-toggle others", title: "Show boxes for the other crops in this photo"}, othersInput, "Others");
      let sourceName = null;
      const hideSpinner = () => { if (spinner.isConnected) spinner.remove(); };
      const applyBoxes = () => {
        if (!frame.classList.contains("full") || !sourceName) return;
        this.paintLightboxBoxes(frame, sourceName, photo.extIds, {
          current: currentInput.checked,
          others: othersInput.checked
        });
      };
      const saveBoxes = () => {
        writeSettings({
          lightboxCurrentBox: currentInput.checked,
          lightboxOtherBoxes: othersInput.checked
        });
        applyBoxes();
      };
      currentInput.addEventListener("change", saveBoxes);
      othersInput.addEventListener("change", saveBoxes);
      const overlay = el("div", {class: "fs-lightbox"},
        el("div", {class: "fs-lb-card"},
          frame,
          el("div", {class: "fs-lb-foot"},
            el("div", {class: "fs-lb-name", text: photo.names.join(", ") || photo.label}),
            el("div", {class: "fs-lb-toggles"}, currentToggle, othersToggle))));
      overlay.addEventListener("pointerdown", event => { if (event.target === overlay) this.closePreview(); });
      document.body.append(overlay);
      this.lightbox = overlay;

      const showOriginal = () => {
        if (this.lightbox !== overlay) return;
        const name = sourceNameFor(this.ds.origUrls, photo.names);
        const url = name && this.ds.origUrls[name];
        if (!url) {
          hideSpinner();
          return;
        }
        const paint = () => {
          if (this.lightbox !== overlay || !img.naturalWidth) return;
          frame.classList.add("full");
          fitLightboxFrame(frame, img);
          sourceName = name;
          applyBoxes();
          hideSpinner();
        };
        img.onload = paint;
        img.onerror = hideSpinner;
        img.src = url;
        if (img.complete && img.naturalWidth) paint();
      };

      if (sourceNameFor(this.ds.origUrls, photo.names)) showOriginal();
      else this.ensureOriginalUrls(photo.names).then(url => {
        if (this.lightbox !== overlay) return;
        if (url) showOriginal();
        else hideSpinner();
      });
    }

    closePreview() {
      if (this.lightbox) this.lightbox.remove();
      this.lightbox = null;
    }

    openOverlay(node, anchor) {
      this.closeOverlay();
      document.body.append(node);
      if (anchor) {
        const r = anchor.getBoundingClientRect();
        node.style.top = (r.bottom + 8) + "px";
        node.style.right = Math.max(8, window.innerWidth - r.right) + "px";
      }
      const outside = event => {
        if (!node.contains(event.target) && event.target !== anchor) this.closeOverlay();
      };
      setTimeout(() => document.addEventListener("pointerdown", outside), 0);
      this.overlay = {node, outside};
    }

    closeOverlay() {
      if (!this.overlay) return;
      document.removeEventListener("pointerdown", this.overlay.outside);
      this.overlay.node.remove();
      this.overlay = null;
    }

    showHelp() {
      const rows = this.cfg.helpRows || [
        ["Drag a photo", "Move it, or the whole selection if it is selected"],
        ["Delete", "Discard the selection"],
        ["Ctrl + Z", "Undo"],
        ["Esc", "Clear selection"],
        ["?", "Show this help"],
        ...(this.cfg.help || [])
      ];
      const node = el("div", {class: "fs-pop fs-help"},
        el("h3", {text: this.cfg.helpTitle || "Shortcuts"}),
        el("p", {class: "fs-muted", text: this.cfg.helpNote || "Drag photos into groups. Delete discards the selection."}),
        el("table", null, rows.map(([k, v]) => el("tr", null, el("td", null, el("kbd", {text: k})), el("td", {text: v})))),
        this.cfg.helpStartOver === false ? null : el("button", {class: "btn", type: "button", text: "Start over", onclick: () => { this.closeOverlay(); if (window.confirm("Clear every group and start over?")) this.startOver(); }}));
      this.openOverlay(node, document.querySelector(".fs-top .btn.icon"));
    }

    /* ----- export ----- */

    buildExport() {
      const s = this.state;
      const groupNameToPaths = {};
      const groupNameToExtIds = {};
      const add = (name, ids, extraNames = []) => {
        const paths = new Set();
        const ext = new Set();
        for (const id of ids) {
          for (const name of this.ds.photos[id].names) paths.add(name);
          for (const e of this.ds.photos[id].extIds) ext.add(e);
        }
        for (const name of extraNames) paths.add(name);
        if (paths.size) {
          groupNameToPaths[name] = [...paths];
          groupNameToExtIds[name] = [...ext];
        }
      };
      if (this.settings.unsorted) add(UNASSIGNED, s.pool, this.ds.skipped);
      const groups = this.store.groups.filter(g => (s.members.get(g.id) || []).length);
      for (const g of groups) add(g.name.trim(), s.members.get(g.id));
      const dup = this.duplicateNames();
      const issues = [];
      if (dup.size) issues.push(`Rename duplicate group names before exporting: ${[...dup].join(", ")}`);
      if (!Object.keys(groupNameToPaths).length) issues.push("Nothing to export yet.");
      return {groupNameToPaths, groupNameToExtIds, groups, issues};
    }

    showExport(anchor) {
      const render = () => {
        const s = this.state;
        const info = this.buildExport();
        const unsorted = el("input", {type: "checkbox"});
        unsorted.checked = this.settings.unsorted;
        unsorted.addEventListener("change", () => {
          this.settings.unsorted = unsorted.checked;
          writeSettings({unsorted: unsorted.checked});
          this.update("settings");
          node.replaceChildren(...render().childNodes);
        });
        const go = linux => {
          const again = this.buildExport();
          if (again.issues.length) { window.alert(again.issues.join("\n")); return; }
          if (this.ds.fromSession && this.ds.sessionId && typeof getBearer === "function") {
            getBearer().then(authorization => fetch(`${API_BASE}/saveSession`, {
              method: "POST",
              headers: {"Authorization": authorization},
              body: JSON.stringify({sessionId: this.ds.sessionId, groupNameToExtIds: again.groupNameToExtIds})
            })).catch(err => console.error(err));
          }
          generateAndDownloadScript(again.groupNameToPaths, linux, this.ds.sessionName);
          this.closeOverlay();
          const folders = Object.keys(again.groupNameToPaths).length;
          this.toast(`Downloaded ${linux ? ".sh" : ".cmd"} script for ${plural(folders, "folder")}`);
        };
        const unsortedLine = this.settings.unsorted
          ? `${plural(s.counts.pool, "unsorted photo")} → Unassigned`
          : `${plural(s.counts.pool, "unsorted photo")} skipped`;
        return el("div", null,
          el("h3", {text: "Export script"}),
          el("ul", {class: "fs-summary"},
            el("li", {text: `${plural(info.groups.length, "group")} · ${plural(s.counts.sorted, "photo")}`}),
            el("li", {text: unsortedLine}),
            el("li", {text: `${plural(s.counts.trash, "discarded photo")} skipped`})),
          el("label", {class: "fs-check"}, unsorted, "Include unsorted photos in Unassigned"),
          info.issues.length ? el("div", {class: "fs-issues"}, info.issues.map(t => el("div", {text: t}))) : null,
          el("div", {class: "fs-export-buttons"},
            el("button", {class: "btn primary", type: "button", text: "Windows (.cmd)", onclick: () => go(false)}),
            el("button", {class: "btn primary", type: "button", text: "Mac / Linux (.sh)", onclick: () => go(true)})),
          el("p", {class: "fs-muted", text: "Download a script that copies each photo into a folder named after its group. Run it from the folder with your originals."}));
      };
      const node = el("div", {class: "fs-pop fs-export"});
      node.append(...render().childNodes);
      this.openOverlay(node, anchor);
    }
  }

  function fingerprint(ds) {
    let h = 2166136261;
    for (const photo of ds.photos) {
      for (const e of photo.extIds) {
        h ^= e + 1;
        h = Math.imul(h, 16777619);
      }
      h ^= 0x9e37;
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0).toString(36) + "-" + ds.n;
  }

  function shQuote(value) {
    return "'" + String(value).replace(/'/g, `'\\''`) + "'";
  }

  function cmdQuote(value) {
    return '"' + String(value).replace(/%/g, "%%").replace(/"/g, '""') + '"';
  }

  function generateAndDownloadScript(groupNameToPaths, targetLinux, sessionName) {
    const sessionNameStd = (sessionName || "").replace(/[^a-zA-Z0-9]/g, "");
    const pauseLine = targetLinux ? `read -rsn1 -p "Press any key to continue . . ."; echo\n` : "pause\n";
    const groups = Object.entries(groupNameToPaths);
    const allPaths = [...new Set(groups.flatMap(([, paths]) => paths))];

    let script = "";
    if (targetLinux) {
      script += "#!/bin/bash\n";
      script += "set -e\n";
    } else {
      script += "@echo off\n";
      script += "setlocal\n";
    }
    script += "\n";
    script += "echo ------------------------------------------\n";
    script += "echo                  FotoSort                 \n";
    script += "echo ------------------------------------------\n";
    if (sessionName) {
      script += targetLinux
        ? `echo Session Name: ${shQuote(sessionName)}\n`
        : `echo Session Name: ${cmdQuote(sessionName)}\n`;
    }
    script += pauseLine;
    script += "\n";
    script += "echo Checking directory\n";
    if (targetLinux) {
      for (const path of allPaths) {
        const q = shQuote(path);
        script += `if [ ! -f ${q} ]; then\n`;
        script += `  echo "ABORTED: Image ${q} was not found. Confirm that this script is in the image directory."\n`;
        script += "  echo No changes made.\n";
        script += "  " + pauseLine;
        script += "  exit 1\n";
        script += "fi\n";
      }
    } else {
      for (const path of allPaths) {
        const q = cmdQuote(path);
        script += `if not exist ${q} (\n`;
        script += `  echo ABORTED: Image ${q} was not found. Confirm that this script is in the image directory.\n`;
        script += "  echo No changes made.\n";
        script += "  " + pauseLine;
        script += "  exit /b 1\n";
        script += ")\n";
      }
    }
    script += "\n";
    script += "echo Creating group folders\n";
    for (const [groupName] of groups) {
      if (targetLinux) {
        script += `mkdir -p ${shQuote(groupName)}\n`;
      } else {
        const q = cmdQuote(groupName);
        script += `if not exist ${q} mkdir ${q}\n`;
      }
    }
    script += "\n";
    script += "echo Copying images to group folders\n";
    for (const [groupName, paths] of groups) {
      for (const path of paths) {
        if (targetLinux) {
          script += `if ! cp ${shQuote(path)} ${shQuote(groupName + "/")}; then\n`;
          script += `  echo "ABORTED: Failed to copy ${shQuote(path)} to ${shQuote(groupName)}."\n`;
          script += "  " + pauseLine;
          script += "  exit 1\n";
          script += "fi\n";
        } else {
          // No trailing backslash before the closing quote — that breaks CMD parsing.
          script += `copy /Y ${cmdQuote(path)} ${cmdQuote(groupName)} >nul\n`;
          script += "if errorlevel 1 (\n";
          script += `  echo ABORTED: Failed to copy ${cmdQuote(path)} to ${cmdQuote(groupName)}.\n`;
          script += "  " + pauseLine;
          script += "  exit /b 1\n";
          script += ")\n";
        }
      }
    }
    script += "\n";
    script += "echo Finished\n";
    if (!targetLinux) script += "endlocal\n";
    script += pauseLine;

    const blob = new Blob([script], {type: "text/plain"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `copy-script${sessionNameStd ? "-" + sessionNameStd : ""}.${targetLinux ? "sh" : "cmd"}`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 2000);
    return script;
  }

  /* ---------- animation ---------- */

  function flip(getNodes, mutate, duration = 240) {
    const before = new Map();
    for (const node of getNodes()) before.set(node, node.getBoundingClientRect());
    mutate();
    const moves = [];
    for (const node of getNodes()) {
      const b = before.get(node);
      if (!b) continue;
      const a = node.getBoundingClientRect();
      const dx = b.left - a.left;
      const dy = b.top - a.top;
      if (Math.abs(dx) >= 1 || Math.abs(dy) >= 1) moves.push([node, dx, dy]);
    }
    for (const [node, dx, dy] of moves) {
      node.animate([{transform: `translate(${dx}px, ${dy}px)`}, {transform: "translate(0, 0)"}],
        {duration, easing: "cubic-bezier(.2, .7, .2, 1)"});
    }
  }

  async function start(cfg) {
    const loaderText = document.querySelector("[data-fs-loader-text]");
    let ds;
    try {
      ds = await loadDataset(cfg);
    } catch (err) {
      console.error(err);
      const versionError = err && typeof err.message === "string" && err.message.startsWith("unsupported data.bin format version");
      if (loaderText) loaderText.textContent = versionError ? err.message : "Could not load photos";
      return null;
    }
    const app = new App(ds, cfg);
    const restored = app.restoreProgress();
    const loader = document.querySelector("[data-fs-loader]");
    if (loader) loader.remove();
    document.getElementById("app").hidden = false;
    app.mountChrome();
    app.enableKeys();
    app.compute();
    cfg.mount(app);
    app.update("init");
    if (restored) {
      app.toast(`Restored your saved progress (${app.state.counts.sorted} sorted)`, {action: {label: "Start over", fn: () => app.startOver()}});
    }
    window.fotosort = app;
    return app;
  }

  return {POOL, TRASH, el, nodes, plural, flip, start, sanitizeName, generateAndDownloadScript};
})();
