// fingerprint.js
(function() {
  'use strict';

  // ============================================================
  // Constante globale pentru procesare și matching
  // ============================================================
  const BLOCK_SIZE = 16;                     // dimensiunea blocului pentru orientare/mască/frecvență
  const MAX_WORK_SIZE = 500;                 // latura maximă a imaginii de lucru (px)
  const FOREGROUND_STD_THRESHOLD = 10;       // prag deviație standard pentru masca foreground
  const ADAPTIVE_WINDOW = 15;                // dimensiunea ferestrei locale pentru binarizare adaptivă
  const ADAPTIVE_K = 0.5;                    // factorul de corecție pentru pragul local
  const MINUTIA_MIN_DIST_BASE = 8;           // distanță de bază minimă între minuții (px)
  const MAX_MINUTIAE = 100;                  // numărul maxim de minuții extrase
  const MATCH_DISTANCE_THRESHOLD = 12;       // prag distanță (px) pentru perechi potrivite
  const MATCH_ANGLE_THRESHOLD = 20 * Math.PI / 180; // prag unghi (radiani) pentru perechi potrivite
  const MATCH_SIGMA_DIST = 5;                // sigma pentru ponderare distanță
  const MATCH_SIGMA_ANGLE = 10 * Math.PI / 180; // sigma pentru ponderare unghi
  const MAX_MATCH_CANDIDATES = 60;           // limităm perechile candidat pentru performanță
  const PRUNE_MIN_LENGTH = 8;                // lungime minimă a ramurilor păstrate în schelet

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

    // Netezirea câmpului de orientare (filtru vectorial circular)
    smoothOrientationField(orient, blockW, blockH);

    return { orient, blockW, blockH };
  }

  /**
   * Netezește câmpul de orientare folosind o fereastră 3x3.
   * Unghiurile sunt tratate modulo π, deci folosim vectori (cos, sin).
   * @param {Float32Array} orient - vectorul de orientări
   * @param {number} blockW
   * @param {number} blockH
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
          // Normalizare la [0, π)
          smoothed[by * blockW + bx] = (avgAngle < 0) ? avgAngle + Math.PI : avgAngle;
        } else {
          smoothed[by * blockW + bx] = orient[by * blockW + bx];
        }
      }
    }

    // Copiem rezultatul înapoi
    for (let i = 0; i < orient.length; i++) {
      orient[i] = smoothed[i];
    }
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
  // Pasul 6: Binarizare adaptivă îmbunătățită
  // ============================================================
  // Folosim o fereastră locală de 15x15 pentru a calcula media și
  // deviația standard. Pragul = media - ADAPTIVE_K * std.
  // Calcul rapid folosind imaginea integrală pentru sumă și sumă de pătrate.

  function computeIntegralImages(gray, w, h) {
    const integral = new Float64Array((w + 1) * (h + 1));
    const integralSq = new Float64Array((w + 1) * (h + 1));

    for (let y = 1; y <= h; y++) {
      for (let x = 1; x <= w; x++) {
        const idx = (y - 1) * w + (x - 1);
        const v = gray[idx];
        const iPrev = (y - 1) * (w + 1) + x;
        const jPrev = y * (w + 1) + (x - 1);
        const ijPrev = (y - 1) * (w + 1) + (x - 1);
        const iCurr = y * (w + 1) + x;

        integral[iCurr] = integral[iPrev] + integral[jPrev] - integral[ijPrev] + v;
        integralSq[iCurr] = integralSq[iPrev] + integralSq[jPrev] - integralSq[ijPrev] + v * v;
      }
    }

    return { integral, integralSq, width: w + 1, height: h + 1 };
  }

  function adaptiveBinarize(gray, maskField, w, h, blockSize) {
    const { mask } = maskField;
    const { integral, integralSq, width: iw, height: ih } = computeIntegralImages(gray, w, h);
    const binary = new Uint8Array(w * h);
    const windowHalf = Math.floor(ADAPTIVE_WINDOW / 2);

    for (let y = 0; y < h; y++) {
      const bx = Math.floor(x => x / blockSize); // placeholder, nu folosim aici
    }

    // Iterăm pixel cu pixel
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        // Determinăm blocul pentru mască
        const blockX = Math.floor(x / blockSize);
        const blockY = Math.floor(y / blockSize);
        // Dacă blocul nu este foreground, rămâne 0
        if (mask[blockY * Math.ceil(w / blockSize) + blockX] === 0) {
          binary[y * w + x] = 0;
          continue;
        }

        // Definim fereastra locală
        const x1 = Math.max(0, x - windowHalf);
        const x2 = Math.min(w - 1, x + windowHalf);
        const y1 = Math.max(0, y - windowHalf);
        const y2 = Math.min(h - 1, y + windowHalf);

        const area = (x2 - x1 + 1) * (y2 - y1 + 1);

        // Coordonate în imaginea integrală (indexate de la 1)
        const ix1 = x1;
        const iy1 = y1;
        const ix2 = x2 + 1;
        const iy2 = y2 + 1;

        const sum = integral[iy2 * iw + ix2] - integral[iy1 * iw + ix2] - integral[iy2 * iw + ix1] + integral[iy1 * iw + ix1];
        const sumSq = integralSq[iy2 * iw + ix2] - integralSq[iy1 * iw + ix2] - integralSq[iy2 * iw + ix1] + integralSq[iy1 * iw + ix1];

        const mean = sum / area;
        const variance = Math.max(0, sumSq / area - mean * mean);
        const std = Math.sqrt(variance);

        const threshold = mean - ADAPTIVE_K * std;
        binary[y * w + x] = gray[y * w + x] < threshold ? 1 : 0;
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
  // Eliminarea ramurilor scurte din schelet (pruning)
  // ============================================================
  /**
   * Parcurge scheletul și elimină ramurile cu lungime mai mică de prag.
   * O ramură este un lanț de pixeli între o terminație (CN=1) și o joncțiune (CN>=3)
   * sau o altă terminație.
   */
  function pruneSkeleton(thinned, w, h, minLength = PRUNE_MIN_LENGTH) {
    const img = new Uint8Array(thinned);
    const visited = new Uint8Array(w * h);

    function getNeighborCoords(x, y) {
      const neigh = [];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h && img[ny * w + nx] === 1) {
            neigh.push([nx, ny]);
          }
        }
      }
      return neigh;
    }

    function countNeighbors(x, y) {
      return getNeighborCoords(x, y).length;
    }

    function isEndPoint(x, y) {
      return countNeighbors(x, y) === 1;
    }

    function isJunction(x, y) {
      return countNeighbors(x, y) >= 3;
    }

    function traceBranch(startX, startY) {
      const branch = [[startX, startY]];
      let curX = startX;
      let curY = startY;
      let prevX = -1;
      let prevY = -1;
      const maxSteps = 50;

      while (branch.length < maxSteps) {
        const neighbors = getNeighborCoords(curX, curY).filter(([nx, ny]) => nx !== prevX || ny !== prevY);
        if (neighbors.length === 0) break;
        const [nextX, nextY] = neighbors[0];
        branch.push([nextX, nextY]);
        prevX = curX;
        prevY = curY;
        curX = nextX;
        curY = nextY;

        if (isEndPoint(curX, curY) || isJunction(curX, curY)) break;
      }

      return branch;
    }

    // Găsim toate terminațiile și pornim trace
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (img[y * w + x] === 1 && !visited[y * w + x] && isEndPoint(x, y)) {
          const branch = traceBranch(x, y);
          if (branch.length < minLength) {
            // Eliminăm ramura (doar dacă nu se termină în alt capăt care e joncțiune importantă)
            let remove = true;
            const [lastX, lastY] = branch[branch.length - 1];
            if (isEndPoint(lastX, lastY) && branch.length >= 2) {
              // Ramura dintre două terminații scurte: eliminăm doar dacă ambele capete sunt terminații
              remove = true;
            } else if (isJunction(lastX, lastY) && branch.length < minLength) {
              // Ramura scurtă atașată la joncțiune: eliminăm, dar păstrăm joncțiunea
              remove = true;
            } else {
              remove = false;
            }

            if (remove) {
              for (const [px, py] of branch) {
                img[py * w + px] = 0;
              }
            }
          }

          // Marcăm ramura ca vizitată indiferent dacă am eliminat sau nu
          for (const [px, py] of branch) {
            visited[py * w + px] = 1;
          }
        }
      }
    }

    return img;
  }

  // ============================================================
  // Pasul 8: Extragerea minuțiilor
  // ============================================================

  function extractMinutiae(thinned, orientField, freqField, maskField, w, h, blockSize) {
    const { orient, blockW, blockH } = orientField;
    const minutiae = [];

    // Folosim coordonate pe schelet
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
          // Determinăm tipul
          const type = cn === 1 ? 'ending' : 'bifurcation';

          // Calculăm unghiul local al crestei din vecinătate
          const angle = getLocalAngle(thinned, x, y, w, h, type);

          minutiae.push({
            x,
            y,
            angle,
            type
          });
        }
      }
    }

    // Filtrarea inițială a minuțiilor false (va fi rafinată mai jos)
    return filterMinutiae(minutiae, maskField, freqField, w, h, blockSize);
  }

  /**
   * Calculează unghiul local al minuției folosind direcția dominantă a crestelor din jur.
   * Pentru terminație, unghiul este direcția de la minuție spre interiorul crestei.
   * Pentru bifurcație, unghiul este media direcțiilor celor trei ramuri.
   */
  function getLocalAngle(thinned, cx, cy, w, h, type) {
    const radius = 8;
    const angles = [];
    const vectors = [];

    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h && thinned[ny * w + nx] === 1) {
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > 0) {
            vectors.push({ dx: dx / dist, dy: dy / dist, dist });
          }
        }
      }
    }

    if (vectors.length === 0) {
      // Fallback la orientarea blocului (nu o avem aici, deci întoarcem 0)
      return 0;
    }

    // Calculăm media vectorială ponderată invers proporțional cu distanța
    let sumX = 0;
    let sumY = 0;
    for (const v of vectors) {
      const weight = 1 / (v.dist + 0.1);
      sumX += v.dx * weight;
      sumY += v.dy * weight;
    }

    let angle = Math.atan2(sumY, sumX);
    // Normalizăm la [0, π)
    if (angle < 0) angle += Math.PI;
    else if (angle >= Math.PI) angle -= Math.PI;

    return angle;
  }

  // ============================================================
  // Pasul 9: Filtrarea minuțiilor false
  // ============================================================

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

    // Calculăm distanța minimă medie dintre creste (1/frecvență)
    // pentru fiecare minuție, folosind frecvența din blocul respectiv
    const getMinDistForMinutia = (x, y) => {
      const bx = Math.floor(x / blockSize);
      const by = Math.floor(y / blockSize);
      if (bx >= 0 && bx < blockW && by >= 0 && by < blockH) {
        const f = freqField[by * blockW + bx];
        if (f > 0) {
          const ridgeDist = 1 / f;
          return Math.max(MINUTIA_MIN_DIST_BASE, 1.5 * ridgeDist);
        }
      }
      return MINUTIA_MIN_DIST_BASE;
    };

    for (const m of minutiae) {
      if (isNearBoundary(m.x, m.y)) continue;

      let tooClose = false;
      const minDist = getMinDistForMinutia(m.x, m.y);
      for (const k of filtered) {
        const dx = m.x - k.x;
        const dy = m.y - k.y;
        if (Math.hypot(dx, dy) < minDist) {
          tooClose = true;
          break;
        }
      }
      if (!tooClose) {
        filtered.push(m);
      }
    }

    // Limităm numărul maxim
    return filtered.length > MAX_MINUTIAE ? filtered.slice(0, MAX_MINUTIAE) : filtered;
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
      angle: p.angle + theta,
      type: p.type
    };
  }

  /**
   * Calculează scorul de potrivire pentru o transformare dată.
   * Întoarce numărul de perechi potrivite și scorul ponderat.
   */
  function evaluateTransformation(listA, listB, theta, refA, refB) {
    const transformed = listA.map(p => transformPoint(p, refA, refB, theta));
    const usedB = new Array(listB.length).fill(false);
    const matches = [];

    for (const tp of transformed) {
      let bestIdx = -1;
      let bestDist = Infinity;
      let bestAngleDiff = Infinity;

      for (let j = 0; j < listB.length; j++) {
        if (usedB[j]) continue;
        const b = listB[j];

        // Verificăm tipul: dacă tipurile diferă, permitem dar cu penalizare
        const typePenalty = tp.type === b.type ? 0 : 0.3; // penalizare la scor

        const d = Math.hypot(tp.x - b.x, tp.y - b.y);
        if (d <= MATCH_DISTANCE_THRESHOLD) {
          const aDiff = angleDiff(tp.angle, b.angle);
          if (aDiff <= MATCH_ANGLE_THRESHOLD) {
            // Dacă găsim o potrivire mai bună (distanță mai mică) o actualizăm
            if (d < bestDist) {
              bestDist = d;
              bestAngleDiff = aDiff;
              bestIdx = j;
            }
          }
        }
      }

      if (bestIdx >= 0) {
        usedB[bestIdx] = true;
        // Calculăm calitatea potrivirii
        const quality = Math.exp(
          - (bestDist * bestDist) / (2 * MATCH_SIGMA_DIST * MATCH_SIGMA_DIST)
          - (bestAngleDiff * bestAngleDiff) / (2 * MATCH_SIGMA_ANGLE * MATCH_SIGMA_ANGLE)
        ) * (tp.type === listB[bestIdx].type ? 1 : 0.7);
        matches.push({ a: tp, b: listB[bestIdx], distance: bestDist, quality });
      }
    }

    const matchedCount = matches.length;
    const totalQuality = matches.reduce((sum, m) => sum + m.quality, 0);
    // Scorul normalizat: calitatea totală / max(1, media numărului de minuții)
    const avgCount = (listA.length + listB.length) / 2;
    const score = totalQuality / Math.max(1, avgCount);

    return { matchedCount, qualityScore: score, matches };
  }

  function matchMinutiae(minA, minB) {
    if (!minA.length || !minB.length) {
      return { score: 0, pairs: [], matchedCount: 0 };
    }

    const listA = minA.slice(0, MAX_MATCH_CANDIDATES);
    const listB = minB.slice(0, MAX_MATCH_CANDIDATES);

    let bestResult = { matchedCount: 0, qualityScore: 0, matches: [] };

    for (const refA of listA) {
      for (const refB of listB) {
        // Încercăm ambele rotații posibile
        const baseTheta = refB.angle - refA.angle;
        const rotations = [baseTheta, baseTheta + Math.PI];

        for (const theta of rotations) {
          const result = evaluateTransformation(listA, listB, theta, refA, refB);
          if (result.matchedCount > bestResult.matchedCount ||
              (result.matchedCount === bestResult.matchedCount && result.qualityScore > bestResult.qualityScore)) {
            bestResult = result;
          }
        }
      }
    }

    // Convertim scorul de calitate într-un procent (0-100)
    const percentage = Math.min(100, bestResult.qualityScore * 100);
    return {
      score: percentage,
      pairs: bestResult.matches,
      matchedCount: bestResult.matchedCount
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

    const freqField = estimateFrequency(normalized, orientField, w, h, BLOCK_SIZE);

    const maskField = segment(normalized, w, h, BLOCK_SIZE);

    const binary = adaptiveBinarize(normalized, maskField, w, h, BLOCK_SIZE);

    let thinned = zhangSuen(binary, w, h);
    thinned = pruneSkeleton(thinned, w, h, PRUNE_MIN_LENGTH);

    const rawMinutiae = extractMinutiae(thinned, orientField, freqField, maskField, w, h, BLOCK_SIZE);
    // extractMinutiae apelează deja filterMinutiae, deci rawMinutiae este deja filtrat
    const minutiae = rawMinutiae;

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