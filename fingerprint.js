// fingerprint.js
(function() {
  'use strict';

  // ============================================================
  // Constante globale pentru procesare și matching
  // ============================================================
  const BLOCK_SIZE = 16;                    // dimensiunea blocurilor pentru orientare/mască/frecvență
  const MAX_WORK_SIZE = 500;                // latura maximă a imaginii de lucru (px)
  const FOREGROUND_STD_THRESHOLD = 12;      // prag deviație standard pentru masca foreground
  const ADAPTIVE_THRESHOLD_DELTA = 8;       // delta sub media locală pentru binarizare adaptivă
  const MINUTIA_MIN_DIST = 8;               // distanță minimă între minuții păstrate
  const MAX_MINUTIAE = 100;                 // numărul maxim de minuții extrase pentru matching
  const MATCH_DISTANCE_THRESHOLD = 12;      // prag distanță (px) pentru perechi potrivite
  const MATCH_ANGLE_THRESHOLD = 20 * Math.PI / 180; // prag unghi (radiani) pentru perechi potrivite
  const MAX_MATCH_CANDIDATES = 60;          // limităm perechile candidat pentru a nu bloca browserul

  // ============================================================
  // Funcții utilitare pentru imagine
  // ============================================================

  /**
   * Încarcă un fișier imagine într-un HTMLImageElement.
   * @param {File} file - fișierul selectat de utilizator.
   * @returns {Promise<HTMLImageElement>}
   */
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

  /**
   * Creează un canvas de lucru redimensionat la maxim 500px pe latura mare.
   * @param {HTMLImageElement} img
   * @returns {HTMLCanvasElement}
   */
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

  /**
   * Extrage pixelii grayscale dintr-un context canvas.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} w
   * @param {number} h
   * @returns {Uint8ClampedArray} - valori 0-255, lungime w*h.
   */
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

    // Dacă imaginea este aproape uniformă, nu amplificăm zgomotul.
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

  /**
   * Scrie imaginea normalizată înapoi într-un canvas, ca ImageData.
   * @param {HTMLCanvasElement} canvas
   * @param {Uint8ClampedArray} gray
   * @param {number} w
   * @param {number} h
   */
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

  /**
   * Calculează gradienții Sobel pe imaginea grayscale.
   * @param {Uint8ClampedArray} gray
   * @param {number} w
   * @param {number} h
   * @returns {{gx: Float32Array, gy: Float32Array}}
   */
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
   * Estimează orientarea dominantă a crestelor per bloc.
   * Unghiul este în radiani, în intervalul [0, π).
   * @param {{gx: Float32Array, gy: Float32Array}} gradients
   * @param {number} w
   * @param {number} h
   * @param {number} blockSize
   * @returns {{orient: Float32Array, blockW: number, blockH: number}}
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

    return { orient, blockW, blockH };
  }

  /**
   * Returnează orientarea blocului care conține punctul (x, y).
   * @param {number} x
   * @param {number} y
   * @param {Float32Array} orient
   * @param {number} blockW
   * @param {number} blockH
   * @param {number} blockSize
   * @returns {number} - unghi în radiani.
   */
  function getOrientationAt(x, y, orient, blockW, blockH, blockSize) {
    const bx = Math.max(0, Math.min(blockW - 1, Math.floor(x / blockSize)));
    const by = Math.max(0, Math.min(blockH - 1, Math.floor(y / blockSize)));
    return orient[by * blockW + bx];
  }

  // ============================================================
  // Pasul 4: Estimarea frecvenței crestelor
  // ============================================================

  /**
   * Netezește un array 1D cu un filtru mobil simplu.
   * @param {Float32Array} arr
   * @param {number} radius
   * @returns {Float32Array}
   */
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

  /**
   * Estimează frecvența medie a crestelor pentru fiecare bloc.
   * Proiectează pixelii blocului pe direcția perpendiculară orientării
   * și măsoară distanța medie dintre minimele locale (crestele întunecate).
   * @param {Uint8ClampedArray} gray
   * @param {{orient: Float32Array, blockW: number, blockH: number}} orientField
   * @param {number} w
   * @param {number} h
   * @param {number} blockSize
   * @returns {Float32Array} - frecvențe per bloc (1/px), valori implicite 0.1.
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

        const smoothed = smooth(proj, 3);

        // Găsim minime locale care corespund crestelor întunecate.
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

  /**
   * Determină blocurile care conțin amprentă, bazat pe deviația standard locală.
   * @param {Uint8ClampedArray} gray
   * @param {number} w
   * @param {number} h
   * @param {number} blockSize
   * @returns {{mask: Uint8Array, blockW: number, blockH: number}}
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
  // Pasul 6: Binarizare adaptivă
  // ============================================================

  /**
   * Binarizează imaginea folosind un prag local per bloc.
   * Păstrează crestele întunecate ca 1 și restul ca 0.
   * @param {Uint8ClampedArray} gray
   * @param {{mask: Uint8Array, blockW: number, blockH: number}} maskField
   * @param {number} w
   * @param {number} h
   * @param {number} blockSize
   * @returns {Uint8Array} - imagine binară (0/1).
   */
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

  /**
   * Aplică algoritmul Zhang-Suen pentru a subția liniile la 1 pixel grosime.
   * Obiectul (crestele) are valoarea 1, fundalul 0.
   * @param {Uint8Array} binary
   * @param {number} w
   * @param {number} h
   * @returns {Uint8Array} - schelet binar.
   */
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

  /**
   * Detectează terminații (CN=1) și bifurcații (CN=3) pe schelet.
   * @param {Uint8Array} thinned
   * @param {{orient: Float32Array, blockW: number, blockH: number}} orientField
   * @param {{mask: Uint8Array, blockW: number, blockH: number}} maskField
   * @param {number} w
   * @param {number} h
   * @param {number} blockSize
   * @returns {Array<{x:number,y:number,angle:number,type:string}>}
   */
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

  /**
   * Elimină minuțiile de pe marginea zonei segmentate și pe cele prea apropiate.
   * @param {Array} minutiae
   * @param {{mask: Uint8Array, blockW: number, blockH: number}} maskField
   * @param {number} w
   * @param {number} h
   * @param {number} blockSize
   * @param {number} minDist
   * @returns {Array} - minuții filtrate, limitate la MAX_MINUTIAE.
   */
  function filterMinutiae(minutiae, maskField, w, h, blockSize, minDist = MINUTIA_MIN_DIST) {
    const { mask, blockW, blockH } = maskField;

    // Verificăm dacă minuția este lângă un bloc de fundal/margine.
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

    // Eliminăm minuțiile foarte apropiate, păstrând una singură.
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

  /**
   * Calculează diferența unghiulară modulo π (orientare de linie, nu vector direcțional).
   * @param {number} a - unghi în radiani
   * @param {number} b - unghi în radiani
   * @returns {number} - diferența în intervalul [0, π/2].
   */
  function angleDiff(a, b) {
    const d = Math.abs(a - b) % Math.PI;
    return d > Math.PI / 2 ? Math.PI - d : d;
  }

  /**
   * Aplică o transformare (rotație + translație) unui punct din setul A.
   * Transformarea aliniază punctul de referință A pe cel de referință B.
   * @param {{x:number,y:number,angle:number}} p - punctul de transformat
   * @param {{x:number,y:number,angle:number}} refA
   * @param {{x:number,y:number,angle:number}} refB
   * @param {number} theta - unghiul de rotație
   * @returns {{x:number,y:number,angle:number}}
   */
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
        const theta = refB.angle - refA.angle;

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

    return {
      score: Math.min(100, bestScore),
      pairs: bestMatches,
      matchedCount: bestMatches.length
    };
  }

  // ============================================================
  // Funcția principală de procesare a unei amprente
  // ============================================================

  /**
   * Procesează un fișier imagine: încărcare, normalizare, orientare,
   * frecvență, segmentare, binarizare, scheletonizare, extragere minuții.
   * @param {File} file
   * @returns {Promise<{canvas:HTMLCanvasElement, width:number, height:number, minutiae:Array}>}
   */
  async function processFile(file) {
    const img = await loadImage(file);
    const canvas = createWorkCanvas(img);
    const w = canvas.width;
    const h = canvas.height;

    // 1-2. Extragere grayscale și normalizare contrast
    const gray = getGrayscale(canvas.getContext('2d'), w, h);
    const normalized = normalizeContrast(gray, w, h);

    // Punem imaginea normalizată în canvas pentru afișare.
    putNormalizedToCanvas(canvas, normalized, w, h);

    // 3. Câmpul de orientare
    const gradients = computeSobel(normalized, w, h);
    const orientField = computeOrientationField(gradients, w, h, BLOCK_SIZE);

    // 4. Frecvența crestelor
    const frequency = estimateFrequency(normalized, orientField, w, h, BLOCK_SIZE);

    // 5. Mască foreground/background
    const maskField = segment(normalized, w, h, BLOCK_SIZE);

    // 6. Binarizare adaptivă
    const binary = adaptiveBinarize(normalized, maskField, w, h, BLOCK_SIZE);

    // 7. Scheletonizare Zhang-Suen
    const thinned = zhangSuen(binary, w, h);

    // 8. Extragerea minuțiilor
    const rawMinutiae = extractMinutiae(thinned, orientField, maskField, w, h, BLOCK_SIZE);

    // 9. Filtrarea minuțiilor false
    const minutiae = filterMinutiae(rawMinutiae, maskField, w, h, BLOCK_SIZE, MINUTIA_MIN_DIST);

    return {
      canvas,
      width: w,
      height: h,
      minutiae,
      // Expunem câteva elemente pentru debugging/educație, fără a fi necesare.
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

  /**
   * Desenează minuțiile peste imaginea deja afișată pe canvas.
   * @param {HTMLCanvasElement} canvas
   * @param {Array} minutiae
   */
  function drawMinutiae(canvas, minutiae) {
    const ctx = canvas.getContext('2d');

    for (const m of minutiae) {
      // Cerc pentru minuție
      ctx.beginPath();
      ctx.arc(m.x, m.y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = m.type === 'ending' ? '#3fd68c' : '#35d0ff';
      ctx.fill();
      ctx.strokeStyle = '#0b1218';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Linie de orientare
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