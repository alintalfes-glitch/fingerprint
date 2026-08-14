// fingerprint.js
// Pipeline complet de comparare amprente pe bază de minuții reale
// (segmentare -> binarizare adaptivă -> scheletonizare Zhang-Suen ->
//  extragere minuții -> matching prin aliniere geometrică).
//
// NOTĂ IMPORTANTĂ: aceasta e o implementare demonstrativă/educațională,
// nu un sistem biometric de nivel forensic sau de securitate reală.
(function () {
  'use strict';

  // ============================================================
  // Constante globale
  // ============================================================
  const BLOCK_SIZE = 16;                       // dimensiunea blocului pentru orientare/mască/frecvență
  const MAX_WORK_SIZE = 500;                   // latura maximă a imaginii de lucru (px)
  const FOREGROUND_STD_THRESHOLD = 8;          // prag deviație standard pentru masca foreground
  const ADAPTIVE_WINDOW = 15;                  // dimensiunea ferestrei locale pentru binarizare adaptivă
  const ADAPTIVE_K = 0.4;                      // factorul de corecție pentru pragul local
  const MINUTIA_MIN_DIST_BASE = 8;             // distanță minimă de bază între minuții (px)
  const MAX_MINUTIAE = 120;                    // numărul maxim de minuții păstrate per amprentă
  const PRUNE_MIN_LENGTH = 8;                  // lungime minimă a ramurilor păstrate în schelet

  // --- Parametri de matching (vezi și notele din matchMinutiae) ---
  const MATCH_DISTANCE_THRESHOLD = 15;         // prag distanță (px) pentru perechi potrivite
  const MATCH_ANGLE_THRESHOLD = 25 * Math.PI / 180; // prag unghi (radiani) pentru perechi potrivite
  const MATCH_SIGMA_DIST = 5;                  // sigma pentru ponderarea distanței (scor)
  const MATCH_SIGMA_ANGLE = 10 * Math.PI / 180; // sigma pentru ponderarea unghiului (scor)
  const MAX_MATCH_CANDIDATES = 80;             // câte minuții (max) participă la SCORARE
  const REF_SEED_COUNT = 25;                   // câte minuții (max) sunt folosite ca ANCORĂ pt. generarea ipotezelor
  const ROTATION_FINE_STEPS = 4;               // pași de rotație fină testați în jurul unghiului de bază
  const ROTATION_FINE_RANGE = 10 * Math.PI / 180; // intervalul de căutare fină (±10°)
  const EARLY_EXIT_RATIO = 0.8;                // oprim căutarea dacă găsim o aliniere care acoperă % din minim(A,B)
  const MAX_EVAL_CALLS = 20000;                // plasă de siguranță: nr. maxim de ipoteze testate indiferent de REF_SEED_COUNT

  // ============================================================
  // Utilitare imagine
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
   * Normalizează contrastul (media și varianța) — variantă simplificată
   * a metodei descrise de Hong, Wan & Jain (1998).
   */
  function normalizeContrast(gray, w, h) {
    const len = w * h;
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

    if (std < 1e-6) return new Uint8ClampedArray(gray);

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
  // Estimarea câmpului de orientare a crestelor
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

  /**
   * Estimează unghiul dominant al crestelor per bloc, folosind formula
   * standard (Hong et al.): direcția gradientului dominant + 90°.
   */
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

    smoothOrientationField(orient, blockW, blockH);
    return { orient, blockW, blockH };
  }

  /**
   * Netezește câmpul de orientare cu o fereastră 3x3, folosind vectori
   * (cos 2θ, sin 2θ) ca să evite artefactele de la periodicitatea mod π.
   */
  function smoothOrientationField(orient, blockW, blockH) {
    const smoothed = new Float32Array(orient.length);
    const neighbors = [
      [-1, -1], [0, -1], [1, -1],
      [-1, 0], [0, 0], [1, 0],
      [-1, 1], [0, 1], [1, 1]
    ];

    for (let by = 0; by < blockH; by++) {
      for (let bx = 0; bx < blockW; bx++) {
        let sumSin = 0;
        let sumCos = 0;
        let count = 0;

        for (const [dx, dy] of neighbors) {
          const nx = bx + dx;
          const ny = by + dy;
          if (nx >= 0 && nx < blockW && ny >= 0 && ny < blockH) {
            const angle = orient[ny * blockW + nx];
            sumSin += Math.sin(2 * angle);
            sumCos += Math.cos(2 * angle);
            count++;
          }
        }

        if (count > 0) {
          const avgAngle = 0.5 * Math.atan2(sumSin, sumCos);
          smoothed[by * blockW + bx] = (avgAngle < 0) ? avgAngle + Math.PI : avgAngle;
        } else {
          smoothed[by * blockW + bx] = orient[by * blockW + bx];
        }
      }
    }

    for (let i = 0; i < orient.length; i++) orient[i] = smoothed[i];
  }

  /** Returnează orientarea blocului care conține punctul (x, y). */
  function getOrientationAt(x, y, orient, blockW, blockH, blockSize) {
    const bx = Math.max(0, Math.min(blockW - 1, Math.floor(x / blockSize)));
    const by = Math.max(0, Math.min(blockH - 1, Math.floor(y / blockSize)));
    return orient[by * blockW + bx];
  }

  // ============================================================
  // Estimarea frecvenței crestelor
  // ============================================================

  function smoothArray(arr, radius) {
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

  /**
   * Estimează, per bloc, distanța medie dintre creste, proiectând
   * intensitatea pe direcția perpendiculară pe orientarea locală și
   * căutând minimele locale ale proiecției (= centrele crestelor).
   */
  function estimateFrequency(gray, orientField, w, h, blockSize) {
    const { orient, blockW, blockH } = orientField;
    const freq = new Float32Array(blockW * blockH).fill(0.1);

    for (let by = 0; by < blockH; by++) {
      for (let bx = 0; bx < blockW; bx++) {
        const angle = orient[by * blockW + bx] + Math.PI / 2; // perpendicular pe creste
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

        const smoothed = smoothArray(proj, 3);

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
          for (let i = 1; i < minima.length; i++) sumDist += minima[i] - minima[i - 1];
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
  // Segmentare foreground/background
  // ============================================================

  /**
   * Marchează blocurile cu deviație standard suficientă drept "amprentă"
   * (foreground); restul e considerat fundal și e exclus din procesare.
   */
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
  // Binarizare adaptivă (prag local pe fereastră 15x15 via imagine integrală)
  // ============================================================
  // NOTĂ: se presupune că liniile crestelor sunt mai închise decât fundalul
  // local (valabil pentru scanări clasice de amprente). La fotografii cu
  // iluminare neobișnuită polaritatea se poate inversa local — dacă apar
  // rezultate ciudate pe poze reale, acesta e primul loc de verificat.

  function computeIntegralImages(gray, w, h) {
    const iw = w + 1;
    const ih = h + 1;
    const integral = new Float64Array(iw * ih);
    const integralSq = new Float64Array(iw * ih);

    for (let y = 1; y <= h; y++) {
      for (let x = 1; x <= w; x++) {
        const idx = (y - 1) * w + (x - 1);
        const v = gray[idx];
        const iCurr = y * iw + x;
        const iPrev = (y - 1) * iw + x;
        const jPrev = y * iw + (x - 1);
        const ijPrev = (y - 1) * iw + (x - 1);

        integral[iCurr] = integral[iPrev] + integral[jPrev] - integral[ijPrev] + v;
        integralSq[iCurr] = integralSq[iPrev] + integralSq[jPrev] - integralSq[ijPrev] + v * v;
      }
    }

    return { integral, integralSq, iw, ih };
  }

  function adaptiveBinarize(gray, maskField, w, h, blockSize) {
    const { mask, blockW } = maskField;
    const { integral, integralSq, iw } = computeIntegralImages(gray, w, h);
    const binary = new Uint8Array(w * h);
    const windowHalf = Math.floor(ADAPTIVE_WINDOW / 2);

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const blockX = Math.floor(x / blockSize);
        const blockY = Math.floor(y / blockSize);
        if (mask[blockY * blockW + blockX] === 0) {
          binary[y * w + x] = 0;
          continue;
        }

        const x1 = Math.max(0, x - windowHalf);
        const x2 = Math.min(w - 1, x + windowHalf);
        const y1 = Math.max(0, y - windowHalf);
        const y2 = Math.min(h - 1, y + windowHalf);

        const area = (x2 - x1 + 1) * (y2 - y1 + 1);

        const ix1 = x1, iy1 = y1, ix2 = x2 + 1, iy2 = y2 + 1;

        const sum = integral[iy2 * iw + ix2] - integral[iy1 * iw + ix2] - integral[iy2 * iw + ix1] + integral[iy1 * iw + ix1];
        const sumSq = integralSq[iy2 * iw + ix2] - integralSq[iy1 * iw + ix2] - integralSq[iy2 * iw + ix1] + integralSq[iy1 * iw + ix1];

        const mean = sum / area;
        const variance = Math.max(0, sumSq / area - mean * mean);
        const std = Math.sqrt(variance);

        const threshold = mean - ADAPTIVE_K * std;
        binary[y * w + x] = gray[y * w + x] < threshold ? 1 : 0;
      }
    }

    return morphologicalClosing(binary, w, h);
  }

  /** Închidere morfologică (dilatare + eroziune, element structural 3x3). */
  function morphologicalClosing(binary, w, h) {
    const dilated = new Uint8Array(w * h);
    const eroded = new Uint8Array(w * h);

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        if (binary[i] === 1) {
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              dilated[(y + dy) * w + (x + dx)] = 1;
            }
          }
        }
      }
    }

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        let allOne = true;
        for (let dy = -1; dy <= 1 && allOne; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dilated[(y + dy) * w + (x + dx)] !== 1) { allOne = false; break; }
          }
        }
        if (allOne) eroded[y * w + x] = 1;
      }
    }

    return eroded;
  }

  // ============================================================
  // Scheletonizare (Zhang-Suen)
  // ============================================================

  function zhangSuenStep(img, w, h, subIter) {
    const marker = new Uint8Array(w * h);
    let changed = false;

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
          const cond = subIter === 0
            ? ((p2 === 0 || p4 === 0 || p6 === 0) && (p4 === 0 || p6 === 0 || p8 === 0))
            : ((p2 === 0 || p4 === 0 || p8 === 0) && (p2 === 0 || p6 === 0 || p8 === 0));
          if (cond) marker[i] = 1;
        }
      }
    }

    for (let i = 0; i < marker.length; i++) {
      if (marker[i]) { img[i] = 0; changed = true; }
    }

    return changed;
  }

  function zhangSuen(binary, w, h) {
    const img = new Uint8Array(binary);
    let step = 0;
    let anyChange = true;

    while (anyChange && step < 100) {
      const changed1 = zhangSuenStep(img, w, h, 0);
      const changed2 = zhangSuenStep(img, w, h, 1);
      anyChange = changed1 || changed2;
      step++;
    }

    return img;
  }

  // ============================================================
  // Eliminarea ramurilor scurte din schelet (pruning)
  // ============================================================

  function pruneSkeleton(thinned, w, h, minLength = PRUNE_MIN_LENGTH) {
    const img = new Uint8Array(thinned);
    const visited = new Uint8Array(w * h);

    function getNeighborCoords(x, y) {
      const neigh = [];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h && img[ny * w + nx] === 1) {
            neigh.push([nx, ny]);
          }
        }
      }
      return neigh;
    }

    function countNeighbors(x, y) { return getNeighborCoords(x, y).length; }
    function isEndPoint(x, y) { return countNeighbors(x, y) === 1; }
    function isJunction(x, y) { return countNeighbors(x, y) >= 3; }

    function traceBranch(startX, startY) {
      const branch = [[startX, startY]];
      let curX = startX, curY = startY, prevX = -1, prevY = -1;
      const maxSteps = 50;

      while (branch.length < maxSteps) {
        const neighbors = getNeighborCoords(curX, curY).filter(([nx, ny]) => nx !== prevX || ny !== prevY);
        if (neighbors.length === 0) break;
        const [nextX, nextY] = neighbors[0];
        branch.push([nextX, nextY]);
        prevX = curX; prevY = curY;
        curX = nextX; curY = nextY;
        if (isEndPoint(curX, curY) || isJunction(curX, curY)) break;
      }

      return branch;
    }

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (img[y * w + x] === 1 && !visited[y * w + x] && isEndPoint(x, y)) {
          const branch = traceBranch(x, y);

          if (branch.length < minLength) {
            for (const [px, py] of branch) img[py * w + px] = 0;
          }

          for (const [px, py] of branch) visited[py * w + px] = 1;
        }
      }
    }

    return img;
  }

  // ============================================================
  // Extragerea minuțiilor (metoda crossing number)
  // ============================================================

  function extractMinutiae(thinned, orientField, freqField, maskField, w, h, blockSize) {
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
        for (let k = 0; k < 8; k++) cn += Math.abs(neighbors[k] - neighbors[k + 1]);
        cn /= 2;

        if (cn === 1 || cn === 3) {
          const type = cn === 1 ? 'ending' : 'bifurcation';
          const angle = getLocalAngle(thinned, x, y, w, h, orient, blockW, blockH, blockSize);
          minutiae.push({ x, y, angle, type });
        }
      }
    }

    return filterMinutiae(minutiae, maskField, freqField, w, h, blockSize);
  }

  /**
   * Unghiul local al minuției: media ponderată a direcțiilor către
   * pixelii de schelet din vecinătate; dacă vecinătatea e prea săracă
   * (posibil aproape de margine), cade pe orientarea blocului.
   */
  function getLocalAngle(thinned, cx, cy, w, h, orient, blockW, blockH, blockSize) {
    const radius = 8;
    let sumX = 0, sumY = 0, count = 0;

    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = cx + dx, ny = cy + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h && thinned[ny * w + nx] === 1) {
          const dist = Math.sqrt(dx * dx + dy * dy);
          const weight = 1 / (dist + 0.1);
          sumX += (dx / dist) * weight;
          sumY += (dy / dist) * weight;
          count++;
        }
      }
    }

    if (count >= 3) {
      let angle = Math.atan2(sumY, sumX);
      if (angle < 0) angle += Math.PI;
      else if (angle >= Math.PI) angle -= Math.PI;
      return angle;
    }

    return getOrientationAt(cx, cy, orient, blockW, blockH, blockSize);
  }

  /** Elimină minuțiile de lângă marginea zonei segmentate și duplicatele apropiate. */
  function filterMinutiae(minutiae, maskField, freqField, w, h, blockSize) {
    const { mask, blockW, blockH } = maskField;
    const filtered = [];

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

    const getMinDistForMinutia = (x, y) => {
      const bx = Math.floor(x / blockSize);
      const by = Math.floor(y / blockSize);
      if (bx >= 0 && bx < blockW && by >= 0 && by < blockH) {
        const f = freqField[by * blockW + bx];
        if (f > 0) return Math.max(MINUTIA_MIN_DIST_BASE, 1.5 * (1 / f));
      }
      return MINUTIA_MIN_DIST_BASE;
    };

    for (const m of minutiae) {
      if (isNearBoundary(m.x, m.y)) continue;

      let tooClose = false;
      const minDist = getMinDistForMinutia(m.x, m.y);
      for (const k of filtered) {
        if (Math.hypot(m.x - k.x, m.y - k.y) < minDist) { tooClose = true; break; }
      }
      if (!tooClose) filtered.push(m);
    }

    return filtered.length > MAX_MINUTIAE ? filtered.slice(0, MAX_MINUTIAE) : filtered;
  }

  // ============================================================
  // Matching prin aliniere geometrică (optimizat)
  // ============================================================
  //
  // Strategie: pentru fiecare pereche (ancoră A, ancoră B) din câte un
  // subset restrâns ("seed") al celor două seturi de minuții, calculăm
  // transformarea (rotație + translație) care ar suprapune ancorele, apoi
  // verificăm câte minuții din SETUL COMPLET A cad, după transformare,
  // suficient de aproape de o minuție din SETUL COMPLET B.
  //
  // Optimizări față de o variantă brute-force:
  //  1. Index spațial (grid) pe listB -> căutarea celui mai apropiat punct
  //     e ~O(1) în loc de O(n), nu O(n) per punct transformat.
  //  2. Ancorele de generare a ipotezelor sunt limitate la REF_SEED_COUNT
  //     (nu toate perechile posibile din seturile complete) -> numărul de
  //     ipoteze testate e independent de câte minuții au fost extrase.
  //  3. Early-exit: ne oprim imediat ce găsim o aliniere care acoperă un
  //     procent mare din minim(|A|, |B|) — nu mai există loc de mai bine.
  //  4. Plasă de siguranță suplimentară: MAX_EVAL_CALLS limitează strict
  //     numărul total de ipoteze indiferent de valorile celorlalți parametri.
  //
  // Fără aceste optimizări, o căutare completă (toate perechile x toate
  // rotațiile x toate scalele) are complexitate O(n^4) în numărul de
  // minuții și devine impracticabilă (minute) la ~80 minuții per amprentă.

  function angleDiff(a, b) {
    const d = Math.abs(a - b) % Math.PI;
    return d > Math.PI / 2 ? Math.PI - d : d;
  }

  function transformPoint(p, refA, refB, theta) {
    const cos = Math.cos(theta), sin = Math.sin(theta);
    const dx = p.x - refA.x;
    const dy = p.y - refA.y;
    return {
      x: refB.x + dx * cos - dy * sin,
      y: refB.y + dx * sin + dy * cos,
      angle: p.angle + theta,
      type: p.type
    };
  }

  function buildSpatialGrid(list, cellSize) {
    const grid = new Map();
    for (let i = 0; i < list.length; i++) {
      const p = list[i];
      const key = Math.floor(p.x / cellSize) + ',' + Math.floor(p.y / cellSize);
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(i);
    }
    return grid;
  }

  function queryNearbyIndices(grid, cellSize, x, y, out) {
    out.length = 0;
    const cx = Math.floor(x / cellSize);
    const cy = Math.floor(y / cellSize);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const bucket = grid.get((cx + dx) + ',' + (cy + dy));
        if (bucket) for (const idx of bucket) out.push(idx);
      }
    }
    return out;
  }

  /**
   * Evaluează o singură ipoteză de transformare (rotație theta în jurul
   * perechii refA/refB) și întoarce câte minuții se potrivesc + un scor
   * ponderat de calitate.
   */
  function evaluateTransformation(listA, listB, theta, refA, refB, gridB, cellSize, scratch) {
    const usedB = scratch.usedB.fill(false, 0, listB.length);
    const matches = [];

    for (let i = 0; i < listA.length; i++) {
      const tp = transformPoint(listA[i], refA, refB, theta);
      const candidates = queryNearbyIndices(gridB, cellSize, tp.x, tp.y, scratch.candBuf);

      let bestIdx = -1, bestDist = Infinity, bestAngleDiff = Infinity;
      for (const j of candidates) {
        if (usedB[j]) continue;
        const b = listB[j];
        const d = Math.hypot(tp.x - b.x, tp.y - b.y);
        if (d <= MATCH_DISTANCE_THRESHOLD) {
          const aDiff = angleDiff(tp.angle, b.angle);
          if (aDiff <= MATCH_ANGLE_THRESHOLD && d < bestDist) {
            bestDist = d; bestAngleDiff = aDiff; bestIdx = j;
          }
        }
      }

      if (bestIdx >= 0) {
        usedB[bestIdx] = true;
        const quality = Math.exp(
          -(bestDist * bestDist) / (2 * MATCH_SIGMA_DIST * MATCH_SIGMA_DIST)
          - (bestAngleDiff * bestAngleDiff) / (2 * MATCH_SIGMA_ANGLE * MATCH_SIGMA_ANGLE)
        ) * (tp.type === listB[bestIdx].type ? 1 : 0.7);
        matches.push({ ax: tp.x, ay: tp.y, bIdx: bestIdx, quality });
      }
    }

    // Filtru de consistență: păstrăm doar potrivirile care au cel puțin
    // o altă potrivire în vecinătate (elimină potriviri izolate/întâmplătoare).
    const filtered = matches.filter((m, idx) => {
      for (let k = 0; k < matches.length; k++) {
        if (k === idx) continue;
        if (Math.hypot(m.ax - matches[k].ax, m.ay - matches[k].ay) < 30) return true;
      }
      return false;
    });

    const totalQuality = filtered.reduce((sum, m) => sum + m.quality, 0);
    const avgCount = (listA.length + listB.length) / 2;

    return {
      matchedCount: filtered.length,
      qualityScore: totalQuality / Math.max(1, avgCount),
      pairs: filtered.map(m => ({ ax: m.ax, ay: m.ay, b: listB[m.bIdx] }))
    };
  }

  /**
   * Compară două seturi de minuții și întoarce un scor de similaritate
   * (0-100) plus lista perechilor de minuții potrivite (pentru afișare).
   */
  function matchMinutiae(minA, minB) {
    if (!minA || !minB || !minA.length || !minB.length) {
      return { score: 0, pairs: [], matchedCount: 0 };
    }

    const listA = minA.slice(0, MAX_MATCH_CANDIDATES);
    const listB = minB.slice(0, MAX_MATCH_CANDIDATES);
    const seedsA = listA.slice(0, REF_SEED_COUNT);
    const seedsB = listB.slice(0, REF_SEED_COUNT);

    const cellSize = MATCH_DISTANCE_THRESHOLD;
    const gridB = buildSpatialGrid(listB, cellSize);
    const scratch = { usedB: new Array(listB.length).fill(false), candBuf: [] };

    let best = { matchedCount: 0, qualityScore: 0, pairs: [] };
    const target = Math.floor(EARLY_EXIT_RATIO * Math.min(listA.length, listB.length));
    let evalCalls = 0;

    outer:
    for (const refA of seedsA) {
      for (const refB of seedsB) {
        const baseTheta = refB.angle - refA.angle;

        // Testăm și baseTheta, și baseTheta+π (ambiguitatea de 180° vine
        // din faptul că unghiul minuției e stocat modulo π).
        for (let variant = 0; variant < 2; variant++) {
          const offset = variant === 0 ? 0 : Math.PI;

          for (let k = -ROTATION_FINE_STEPS; k <= ROTATION_FINE_STEPS; k++) {
            const theta = baseTheta + offset + (ROTATION_FINE_RANGE * k) / ROTATION_FINE_STEPS;

            evalCalls++;
            if (evalCalls > MAX_EVAL_CALLS) break outer; // plasă de siguranță

            const result = evaluateTransformation(listA, listB, theta, refA, refB, gridB, cellSize, scratch);

            if (
              result.matchedCount > best.matchedCount ||
              (result.matchedCount === best.matchedCount && result.qualityScore > best.qualityScore)
            ) {
              best = result;
              if (best.matchedCount >= target) break outer; // aliniere deja foarte bună
            }
          }
        }
      }
    }

    return {
      score: Math.min(100, best.qualityScore * 100),
      pairs: best.pairs,
      matchedCount: best.matchedCount
    };
  }

  // ============================================================
  // Pipeline principal: procesarea unei singure amprente
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
    const freqField = estimateFrequency(normalized, orientField, w, h, BLOCK_SIZE);
    const maskField = segment(normalized, w, h, BLOCK_SIZE);
    const binary = adaptiveBinarize(normalized, maskField, w, h, BLOCK_SIZE);

    let thinned = zhangSuen(binary, w, h);
    thinned = pruneSkeleton(thinned, w, h, PRUNE_MIN_LENGTH);

    const minutiae = extractMinutiae(thinned, orientField, freqField, maskField, w, h, BLOCK_SIZE);

    return {
      canvas,
      width: w,
      height: h,
      minutiae,
      debug: {
        frequency: freqField,
        mask: maskField.mask,
        thinned
      }
    };
  }

  // ============================================================
  // Afișarea minuțiilor pe canvas
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
  const api = {
    processFile,
    matchMinutiae,
    drawMinutiae,
    constants: {
      BLOCK_SIZE,
      MAX_WORK_SIZE,
      MATCH_DISTANCE_THRESHOLD,
      MATCH_ANGLE_THRESHOLD,
      MAX_MATCH_CANDIDATES,
      REF_SEED_COUNT
    }
  };

  if (typeof window !== 'undefined') window.FingerprintProcessor = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
