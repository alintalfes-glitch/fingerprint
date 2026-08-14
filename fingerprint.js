// fingerprint.js
(function() {
  'use strict';

  // ============================================================
  // Constante globale pentru procesare și matching
  // ============================================================
  const BLOCK_SIZE = 16;
  const MAX_WORK_SIZE = 500;
  const FOREGROUND_STD_THRESHOLD = 12;
  const ADAPTIVE_THRESHOLD_DELTA = 8;
  const MINUTIA_MIN_DIST = 8;
  const MAX_MINUTIAE = 100;
  const MATCH_DISTANCE_THRESHOLD = 12;
  const MATCH_ANGLE_THRESHOLD = 20 * Math.PI / 180;
  const MAX_MATCH_CANDIDATES = 60;

  // ============================================================
  // Funcții utilitare pentru imagine
  // ============================================================

  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Nu s-a putut citi imaginea.'));
      };
      img.src = url;
    });
  }

  function createWorkCanvas(img) {
    const maxSize = MAX_WORK_SIZE;
    let w = img.naturalWidth || img.width;
    let h = img.naturalHeight || img.height;
    const scale = Math.min(1, maxSize / Math.max(w, h));
    w = Math.max(1, Math.round(w * scale));
    h = Math.max(1, Math.round(h * scale));

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    return canvas;
  }

  function getGrayscale(ctx, w, h) {
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;
    const gray = new Uint8ClampedArray(w * h);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      gray[p] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    return gray;
  }

  /**
   * Normalizează contrastul (media și varianța) conform metodei Hong et al.
   * @param {Uint8ClampedArray} gray - imagine grayscale 0-255.
   * @param {number} w
   * @param {number} h
   * @returns {Uint8ClampedArray} - imagine normalizată 0-255.
   */
  function normalizeContrast(gray, w, h) {
    const len = w * h;
    // Evităm diviziunea la zero pentru imagini goale (caz extrem)
    if (len === 0) return new Uint8ClampedArray(gray);

    let sum = 0;
    for (let i = 0; i < len; i++) sum += gray[i];

    const mean = sum / len;

    let variance = 0;
    for (let i = 0; i < len; i++) {
      const d = gray[i] - mean;
      variance += d * d;
    }
    variance /= len;
    const std = Math.sqrt(variance);

    if (std < 1e-6) {
      return new Uint8ClampedArray(gray);
    }

    const desiredMean = 128;
    const desiredStd = 60;
    const out = new Uint8ClampedArray(len);
    for (let i = 0; i < len; i++) {
      const val = (gray[i] - mean) * (desiredStd / std) + desiredMean;
      out[i] = Math.max(0, Math.min(255, Math.round(val)));
    }
    return out;
  }

  function putNormalizedToCanvas(canvas, gray, w, h) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(w, h);
    const data = imageData.data;
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      data[p] = gray[i];
      data[p + 1] = gray[i];
      data[p + 2] = gray[i];
      data[p + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }

  // ============================================================
  // Pasul 3: Estimarea câmpului de orientare a crestelor
  // ============================================================

  function computeSobel(gray, w, h) {
    const gx = new Float32Array(w * h);
    const gy = new Float32Array(w * h);

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const topLeft = gray[(y - 1) * w + x - 1];
        const top = gray[(y - 1) * w + x];
        const topRight = gray[(y - 1) * w + x + 1];
        const midLeft = gray[y * w + x - 1];
        const midRight = gray[y * w + x + 1];
        const bottomLeft = gray[(y + 1) * w + x - 1];
        const bottom = gray[(y + 1) * w + x];
        const bottomRight = gray[(y + 1) * w + x + 1];

        gx[i] = (topRight + 2 * midRight + bottomRight) - (topLeft + 2 * midLeft + bottomLeft);
        gy[i] = (bottomLeft + 2 * bottom + bottomRight) - (topLeft + 2 * top + topRight);
      }
    }
    return { gx, gy };
  }

  function computeOrientationField(gradients, w, h, blockSize) {
    const { gx, gy } = gradients;
    const blockW = Math.ceil(w / blockSize);
    const blockH = Math.ceil(h / blockSize);
    const orient = new Float32Array(blockW * blockH);

    for (let by = 0; by < blockH; by++) {
      for (let bx = 0; bx < blockW; bx++) {
        let Vx = 0;
        let Vy = 0;
        const startX = bx * blockSize;
        const startY = by * blockSize;
        const endX = Math.min(w, startX + blockSize);
        const endY = Math.min(h, startY + blockSize);

        for (let y = startY; y < endY; y++) {
          for (let x = startX; x < endX; x++) {
            const i = y * w + x;
            Vx += gx[i] * gx[i] - gy[i] * gy[i];
            Vy += 2 * gx[i] * gy[i];
          }
        }

        let theta = 0.5 * Math.atan2(Vy, Vx) + Math.PI / 2;
        if (theta < 0) theta += Math.PI;
        else if (theta >= Math.PI) theta -= Math.PI;

        orient[by * blockW + bx] = theta;
      }
    }

    return { orient, blockW, blockH };
  }

  function getOrientationAt(x, y, orient, blockW, blockH, blockSize) {
    const bx = Math.max(0, Math.min(blockW - 1, Math.floor(x / blockSize)));
    const by = Math.max(0, Math.min(blockH - 1, Math.floor(y / blockSize)));
    return orient[by * blockW + bx];
  }

  // ============================================================
  // Pasul 4: Estimarea frecvenței crestelor
  // ============================================================

  function smooth(arr, radius) {
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      let sum = 0;
      let n = 0;
      for (let j = -radius; j <= radius; j++) {
        const idx = i + j;
        if (idx >= 0 && idx < arr.length) {
          sum += arr[idx];
          n++;
        }
      }
      out[i] = sum / n;
    }
    return out;
  }

  function estimateFrequency(gray, orientField, w, h, blockSize) {
    const { orient, blockW, blockH } = orientField;
    const freq = new Float32Array(blockW * blockH).fill(0.1);

    for (let by = 0; by < blockH; by++) {
      for (let bx = 0; bx < blockW; bx++) {
        const angle = orient[by * blockW + bx] + Math.PI / 2;
        const cx = (bx + 0.5) * blockSize;
        const cy = (by + 0.5) * blockSize;

        const L = blockSize * 3;
        const bins = 64;
        const proj = new Float32Array(bins);
        const count = new Int32Array(bins);

        const startX = Math.max(0, bx * blockSize - blockSize);
        const startY = Math.max(0, by * blockSize - blockSize);
        const endX = Math.min(w, (bx + 1) * blockSize + blockSize);
        const endY = Math.min(h, (by + 1) * blockSize + blockSize);

        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        for (let y = startY; y < endY; y++) {
          for (let x = startX; x < endX; x++) {
            const dx = x - cx;
            const dy = y - cy;
            const t = dx * cos + dy * sin;
            const bin = Math.floor(((t + L / 2) / L) * bins);
            if (bin >= 0 && bin < bins) {
              proj[bin] += gray[y * w + x];
              count[bin]++;
            }
          }
        }

        for (let b = 0; b < bins; b++) {
          if (count[b] > 0) proj[b] /= count[b];
        }

        const smoothed = smooth(proj, 3);

        const minima = [];
        for (let b = 2; b < bins - 2; b++) {
          if (
            smoothed[b] < smoothed[b - 1] &&
            smoothed[b] < smoothed[b + 1] &&
            smoothed[b] < smoothed[b - 2] &&
            smoothed[b] < smoothed[b + 2]
          ) {
            minima.push(b);
          }
        }

        if (minima.length >= 2) {
          let sumDist = 0;
          for (let i = 1; i < minima.length; i++) {
            sumDist += minima[i] - minima[i - 1];
          }
          const avgDistPixels = (sumDist / (minima.length - 1)) * (L / bins);
          if (avgDistPixels >= 3 && avgDistPixels <= 30) {
            freq[by * blockW + bx] = 1 / avgDistPixels;
          }
        }
      }
    }

    return freq;
  }

  // ============================================================
  // Pasul 5: Segmentare foreground/background
  // ============================================================

  function segment(gray, w, h, blockSize) {
    const blockW = Math.ceil(w / blockSize);
    const blockH = Math.ceil(h / blockSize);
    const mask = new Uint8Array(blockW * blockH);

    for (let by = 0; by < blockH; by++) {
      const startY = by * blockSize;
      const endY = Math.min(h, startY + blockSize);

      for (let bx = 0; bx < blockW; bx++) {
        const startX = bx * blockSize;
        const endX = Math.min(w, startX + blockSize);

        let sum = 0;
        let sumSq = 0;
        let n = 0;

        for (let y = startY; y < endY; y++) {
          for (let x = startX; x < endX; x++) {
            const v = gray[y * w + x];
            sum += v;
            sumSq += v * v;
            n++;
          }
        }

        if (n === 0) {
          mask[by * blockW + bx] = 0;
          continue;
        }

        const mean = sum / n;
        const variance = sumSq / n - mean * mean;
        const std = Math.sqrt(Math.max(0, variance));
        mask[by * blockW + bx] = std > FOREGROUND_STD_THRESHOLD ? 1 : 0;
      }
    }

    return { mask, blockW, blockH };
  }

  // ============================================================
  // Pasul 6: Binarizare adaptivă
  // ============================================================

  function adaptiveBinarize(gray, maskField, w, h, blockSize) {
    const { mask, blockW, blockH } = maskField;
    const binary = new Uint8Array(w * h);

    for (let by = 0; by < blockH; by++) {
      for (let bx = 0; bx < blockW; bx++) {
        if (mask[by * blockW + bx] === 0) continue;

        const startX = bx * blockSize;
        const startY = by * blockSize;
        const endX = Math.min(w, startX + blockSize);
        const endY = Math.min(h, startY + blockSize);

        let sum = 0;
        let n = 0;
        for (let y = startY; y < endY; y++) {
          for (let x = startX; x < endX; x++) {
            sum += gray[y * w + x];
            n++;
          }
        }

        const mean = sum / n;
        const threshold = mean - ADAPTIVE_THRESHOLD_DELTA;

        for (let y = startY; y < endY; y++) {
          for (let x = startX; x < endX; x++) {
            binary[y * w + x] = gray[y * w + x] < threshold ? 1 : 0;
          }
        }
      }
    }

    return binary;
  }

  // ============================================================
  // Pasul 7: Scheletonizare (Zhang-Suen)
  // ============================================================

  function zhangSuen(binary, w, h) {
    let img = new Uint8Array(binary);
    let step = 0;
    let changed = true;

    while (changed && step < 100) {
      changed = false;
      const marker = new Uint8Array(w * h);

      // Prima iterație parțială
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const i = y * w + x;
          if (img[i] !== 1) continue;

          const p2 = img[(y - 1) * w + x];
          const p3 = img[(y - 1) * w + x + 1];
          const p4 = img[y * w + x + 1];
          const p5 = img[(y + 1) * w + x + 1];
          const p6 = img[(y + 1) * w + x];
          const p7 = img[(y + 1) * w + x - 1];
          const p8 = img[y * w + x - 1];
          const p9 = img[(y - 1) * w + x - 1];

          const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          let A = 0;
          if (p2 === 0 && p3 === 1) A++;
          if (p3 === 0 && p4 === 1) A++;
          if (p4 === 0 && p5 === 1) A++;
          if (p5 === 0 && p6 === 1) A++;
          if (p6 === 0 && p7 === 1) A++;
          if (p7 === 0 && p8 === 1) A++;
          if (p8 === 0 && p9 === 1) A++;
          if (p9 === 0 && p2 === 1) A++;

          if (B >= 2 && B <= 6 && A === 1) {
            if ((p2 === 0 || p4 === 0 || p6 === 0) && (p4 === 0 || p6 === 0 || p8 === 0)) {
              marker[i] = 1;
            }
          }
        }
      }

      for (let i = 0; i < marker.length; i++) {
        if (marker[i]) {
          img[i] = 0;
          changed = true;
        }
      }

      if (!changed) break;

      changed = false;
      const marker2 = new Uint8Array(w * h);

      // A doua iterație parțială
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const i = y * w + x;
          if (img[i] !== 1) continue;

          const p2 = img[(y - 1) * w + x];
          const p3 = img[(y - 1) * w + x + 1];
          const p4 = img[y * w + x + 1];
          const p5 = img[(y + 1) * w + x + 1];
          const p6 = img[(y + 1) * w + x];
          const p7 = img[(y + 1) * w + x - 1];
          const p8 = img[y * w + x - 1];
          const p9 = img[(y - 1) * w + x - 1];

          const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          let A = 0;
          if (p2 === 0 && p3 === 1) A++;
          if (p3 === 0 && p4 === 1) A++;
          if (p4 === 0 && p5 === 1) A++;
          if (p5 === 0 && p6 === 1) A++;
          if (p6 === 0 && p7 === 1) A++;
          if (p7 === 0 && p8 === 1) A++;
          if (p8 === 0 && p9 === 1) A++;
          if (p9 === 0 && p2 === 1) A++;

          if (B >= 2 && B <= 6 && A === 1) {
            if ((p2 === 0 || p4 === 0 || p8 === 0) && (p2 === 0 || p6 === 0 || p8 === 0)) {
              marker2[i] = 1;
            }
          }
        }
      }

      for (let i = 0; i < marker2.length; i++) {
        if (marker2[i]) {
          img[i] = 0;
          changed = true;
        }
      }

      step++;
    }

    return img;
  }

  // ============================================================
  // Pasul 8: Extragerea minuțiilor
  // ============================================================

  function extractMinutiae(thinned, orientField, maskField, w, h, blockSize) {
    const { orient, blockW, blockH } = orientField;
    const minutiae = [];

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        if (thinned[i] !== 1) continue;

        const p2 = thinned[(y - 1) * w + x];
        const p3 = thinned[(y - 1) * w + x + 1];
        const p4 = thinned[y * w + x + 1];
        const p5 = thinned[(y + 1) * w + x + 1];
        const p6 = thinned[(y + 1) * w + x];
        const p7 = thinned[(y + 1) * w + x - 1];
        const p8 = thinned[y * w + x - 1];
        const p9 = thinned[(y - 1) * w + x - 1];

        const neighbors = [p2, p3, p4, p5, p6, p7, p8, p9, p2];
        let cn = 0;
        for (let k = 0; k < 8; k++) {
          cn += Math.abs(neighbors[k] - neighbors[k + 1]);
        }
        cn /= 2;

        if (cn === 1 || cn === 3) {
          const angle = getOrientationAt(x, y, orient, blockW, blockH, blockSize);
          minutiae.push({
            x,
            y,
            angle,
            type: cn === 1 ? 'ending' : 'bifurcation'
          });
        }
      }
    }

    return minutiae;
  }

  // ============================================================
  // Pasul 9: Filtrarea minuțiilor false
  // ============================================================

  function filterMinutiae(minutiae, maskField, w, h, blockSize, minDist = MINUTIA_MIN_DIST) {
    const { mask, blockW, blockH } = maskField;

    const isNearBoundary = (x, y) => {
      const bx = Math.floor(x / blockSize);
      const by = Math.floor(y / blockSize);

      for (let ny = by - 1; ny <= by + 1; ny++) {
        for (let nx = bx - 1; nx <= bx + 1; nx++) {
          if (nx < 0 || ny < 0 || nx >= blockW || ny >= blockH) return true;
          if (mask[ny * blockW + nx] === 0) return true;
        }
      }
      return false;
    };

    const filtered = minutiae.filter(m => !isNearBoundary(m.x, m.y));

    const kept = [];
    for (const m of filtered) {
      let tooClose = false;
      for (const k of kept) {
        const dx = m.x - k.x;
        const dy = m.y - k.y;
        if (Math.hypot(dx, dy) < minDist) {
          tooClose = true;
          break;
        }
      }
      if (!tooClose) kept.push(m);
    }

    return kept.length > MAX_MINUTIAE ? kept.slice(0, MAX_MINUTIAE) : kept;
  }

  // ============================================================
  // Algoritmul de matching bazat pe aliniere
  // ============================================================

  function angleDiff(a, b) {
    const d = Math.abs(a - b) % Math.PI;
    return d > Math.PI / 2 ? Math.PI - d : d;
  }

  function transformPoint(p, refA, refB, theta) {
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const dx = p.x - refA.x;
    const dy = p.y - refA.y;
    return {
      x: refB.x + dx * cos - dy * sin,
      y: refB.y + dx * sin + dy * cos,
      angle: p.angle + theta
    };
  }

  /**
   * Găsește cea mai bună aliniere și calculează scorul de similaritate.
   * @param {Array} minA - minuții set A
   * @param {Array} minB - minuții set B
   * @returns {{score:number, pairs:Array, matchedCount:number}}
   */
  function matchMinutiae(minA, minB) {
    if (!minA.length || !minB.length) {
      return { score: 0, pairs: [], matchedCount: 0 };
    }

    const listA = minA.slice(0, MAX_MATCH_CANDIDATES);
    const listB = minB.slice(0, MAX_MATCH_CANDIDATES);

    let bestMatches = [];
    let bestScore = 0;

    for (const refA of listA) {
      for (const refB of listB) {
        // Diferența de unghi de bază (modulo π)
        const baseTheta = refB.angle - refA.angle;

        // Încercăm ambele rotații posibile: θ și θ+π
        // deoarece orientarea crestelor este definită doar modulo π.
        const rotations = [baseTheta, baseTheta + Math.PI];

        for (const theta of rotations) {
          const transformed = listA.map(p => transformPoint(p, refA, refB, theta));
          const usedB = new Array(listB.length).fill(false);
          const matches = [];

          for (const tp of transformed) {
            let bestIdx = -1;
            let bestDist = MATCH_DISTANCE_THRESHOLD;

            for (let j = 0; j < listB.length; j++) {
              if (usedB[j]) continue;
              const b = listB[j];

              const d = Math.hypot(tp.x - b.x, tp.y - b.y);
              if (d <= bestDist && angleDiff(tp.angle, b.angle) <= MATCH_ANGLE_THRESHOLD) {
                bestDist = d;
                bestIdx = j;
              }
            }

            if (bestIdx >= 0) {
              usedB[bestIdx] = true;
              matches.push({
                a: tp,
                b: listB[bestIdx],
                distance: bestDist
              });
            }
          }

          const matchedCount = matches.length;
          const totalA = minA.length;
          const totalB = minB.length;
          const score = (matchedCount / ((totalA + totalB) / 2)) * 100;

          if (matchedCount > bestMatches.length) {
            bestMatches = matches;
            bestScore = score;
          }
        }
      }
    }

    return {
      score: Math.min(100, bestScore),
      pairs: bestMatches,
      matchedCount: bestMatches.length
    };
  }

  // ============================================================
  // Funcția principală de procesare a unei amprente
  // ============================================================

  async function processFile(file) {
    const img = await loadImage(file);
    const canvas = createWorkCanvas(img);
    const w = canvas.width;
    const h = canvas.height;

    const gray = getGrayscale(canvas.getContext('2d'), w, h);
    const normalized = normalizeContrast(gray, w, h);

    putNormalizedToCanvas(canvas, normalized, w, h);

    const gradients = computeSobel(normalized, w, h);
    const orientField = computeOrientationField(gradients, w, h, BLOCK_SIZE);

    const frequency = estimateFrequency(normalized, orientField, w, h, BLOCK_SIZE);

    const maskField = segment(normalized, w, h, BLOCK_SIZE);

    const binary = adaptiveBinarize(normalized, maskField, w, h, BLOCK_SIZE);

    const thinned = zhangSuen(binary, w, h);

    const rawMinutiae = extractMinutiae(thinned, orientField, maskField, w, h, BLOCK_SIZE);

    const minutiae = filterMinutiae(rawMinutiae, maskField, w, h, BLOCK_SIZE, MINUTIA_MIN_DIST);

    return {
      canvas,
      width: w,
      height: h,
      minutiae,
      debug: {
        frequency,
        mask: maskField.mask,
        thinned
      }
    };
  }

  // ============================================================
  // Funcție pentru afișarea minuțiilor pe canvas
  // ============================================================

  function drawMinutiae(canvas, minutiae) {
    const ctx = canvas.getContext('2d');

    for (const m of minutiae) {
      ctx.beginPath();
      ctx.arc(m.x, m.y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = m.type === 'ending' ? '#3fd68c' : '#35d0ff';
      ctx.fill();
      ctx.strokeStyle = '#0b1218';
      ctx.lineWidth = 1;
      ctx.stroke();

      const len = 8;
      ctx.beginPath();
      ctx.moveTo(m.x, m.y);
      ctx.lineTo(m.x + Math.cos(m.angle) * len, m.y + Math.sin(m.angle) * len);
      ctx.strokeStyle = '#ffb454';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // ============================================================
  // API public
  // ============================================================
  window.FingerprintProcessor = {
    processFile,
    matchMinutiae,
    drawMinutiae,
    constants: {
      BLOCK_SIZE,
      MAX_WORK_SIZE,
      MATCH_DISTANCE_THRESHOLD,
      MATCH_ANGLE_THRESHOLD
    }
  };
})();