function uncondenseMatrix(condensedMatrix, size) {
  let matrix = Array.from({length: size}, () => Array(size).fill(0));
  let idx = 0;
  for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) {
          matrix[i][j] = condensedMatrix[idx];
          matrix[j][i] = condensedMatrix[idx];
          idx++;
      }
  }
  return matrix;
}


class Clusterer {
  constructor(condensedMatrix, n) {
    if (!Array.isArray(condensedMatrix)) throw new Error();
    this.distanceMatrix = uncondenseMatrix(condensedMatrix, n);
  };

  runSingleLinkage(epsilon, minClusterSize = 1) {
    const n = this.distanceMatrix.length;
    let labels = new Array(n).fill(-1);

    let newLabel = 0;
    for (let i = 0; i < n; i++) {
      if (labels[i] !== -1) continue
      let queue = [i];
      let cluster = [];
      while (queue.length > 0) {
        const q = queue.shift();
        if (labels[q] === newLabel) continue;
        labels[q] = newLabel;
        cluster.push(q);
        for (let j = i; j < n; j++) {  // add neighbors within epsilon
          if (this.distanceMatrix[q][j] <= epsilon && labels[j] === -1) {
            queue.push(j);
          }
        }
      }

      if (cluster.length >= minClusterSize) {
        newLabel++;
      } else {
        cluster.forEach(q => {labels[q] = -1;});  // revert label
      }
    }
    return labels;
  }

  runKMedoids(k, minClusterSize = 1) {
    const n = this.distanceMatrix.length;
    let labels = new Array(n).fill(0);
    if (k <= 1) return labels;
    if (k > n) k = n;

    let medoids;
    if (k === 2) {  // select furthest medoids
      medoids = findMaxIndices(this.distanceMatrix);
    } else {  // equally sample medoids
      medoids = Array.from({length: k}, (_, i) => Math.round(i * (n - 1) / (k - 1)));
    }

    let iter = 0;
    let maxIter = n > 10 ? n : 10;
    let medoidChanged = true;
    while (medoidChanged) {
      medoidChanged = false;
      if (iter > maxIter) {
        alert("undefined clustering error");  // should never happen
        return labels;
      }
      iter++;

      // assign each point to the nearest medoid
      for (let i = 0; i < n; i++) {  // each point
        let minDist = Infinity;
        for (let m = 0; m < k; m++) {  // each medoid
          const dist = this.distanceMatrix[i][medoids[m]];
          if (dist < minDist) {
            minDist = dist;
            labels[i] = m;
          }
        }
      }

      // update medoids
      for (let m = 0; m < k; m++) {
        let clusterIdx = labels.map((label, idx) => (label === m ? idx : null)).filter(idx => idx !== null);
        let minSumDist = Infinity;
        let newMedoid = medoids[m];
        for (let i of clusterIdx) {
          let sumDist = 0;
          for (let j of clusterIdx) {
            sumDist += this.distanceMatrix[i][j];
          }
          if (sumDist < minSumDist) {
            minSumDist = sumDist;
            newMedoid = i;
          }
        }
        if (newMedoid !== medoids[m]) {
          medoids[m] = newMedoid;
          medoidChanged = true;
        }
      }
    }
    labels = condenseLabels(labels, minClusterSize)
    return labels;
  }

  runKMedoidsSubset(k, subsetIdx, minClusterSize = 1) {
    const origDistanceMatrix = this.distanceMatrix;
    this.distanceMatrix = subsetIdx.map(i => subsetIdx.map(j => this.distanceMatrix[i][j]));
    const labels = this.runKMedoids(k, minClusterSize);
    this.distanceMatrix = origDistanceMatrix;
    return labels;
  }
}

function findMaxIndices(distanceMatrix) {
  let maxVal = -Infinity;
  let maxIndices = [-1, -1];
  for (let i = 0; i < distanceMatrix.length; i++) {
    for (let j = i + 1; j < distanceMatrix[i].length; j++) {  // skip lower triangle
      if (distanceMatrix[i][j] > maxVal) {
        maxVal = distanceMatrix[i][j];
        maxIndices = [i, j];
      }
    }
  }
  return maxIndices;
}

function findMaxIndicesSubset(distanceMatrix, subsetIdx) {
  if (subsetIdx.length === 1) return [0, 0];
  if (subsetIdx.length === 2) return [0, 1];
  let maxVal = -Infinity;
  let maxIndices = [-1, -1];
  for (let x = 0; x < subsetIdx.length; x++) {
    for (let y = x + 1; y < subsetIdx.length; y++) {  // skip lower triangle
      const i = subsetIdx[x];
      const j = subsetIdx[y];
      if (distanceMatrix[i][j] > maxVal) {
        maxVal = distanceMatrix[i][j];
        maxIndices = [i, j];
      }
    }
  }

  let avgDistIdx0 = 0, avgDistIdx1 = 0;
  for (const i of subsetIdx) {
    avgDistIdx0 += distanceMatrix[i][maxIndices[0]];
    avgDistIdx1 += distanceMatrix[i][maxIndices[1]];
  }
  avgDistIdx0 /= subsetIdx.length;
  avgDistIdx1 /= subsetIdx.length;
  if (avgDistIdx0 > avgDistIdx1) {  // most similar to least similar
    return [maxIndices[1], maxIndices[0]];
  }
  return maxIndices;
}

function condenseLabels(labels, minSamples) {
  let labelCounts = {};
  labels.forEach(label => {
    labelCounts[label] = (labelCounts[label] || 0) + 1;
  });
  let updatedLabels = labels.map(num => labelCounts[num] < minSamples ? -1 : num);

  // remap unused labels
  const uniqueClasses = [...new Set(updatedLabels)].sort();
  let offset = uniqueClasses[0] === -1 ? -1 : 0;

  const classMapping = {};
  uniqueClasses.forEach((cls, index) => {
    classMapping[cls] = index + offset;
  });
  updatedLabels = updatedLabels.map(cls => classMapping[cls]);

  return updatedLabels;
}
