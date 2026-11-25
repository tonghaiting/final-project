/* 把用户导入的 STL / OBJ / 图片栅格化成二维轮廓，供求解器做浸没固体。
 *
 * 3D 模型的处理方式：沿选定的轴做正投影，得到"流体看到"的二维剪影。
 * 所有三角形统一成正绕向后一次 nonzero 填充，重叠的背面不会把正面抵消掉。
 * 输出是一张裁到内容边界的透明底白图，面板再按网格尺寸缩放成掩膜。
 *
 * 普通脚本（非 ES module），file:// 直接打开也能用。
 */
(function (global) {
  'use strict';

  var RES = 512;
  var PAD = 0.03;

  function parseSTL(buf) {
    var dv = new DataView(buf);
    if (buf.byteLength >= 84) {
      var n = dv.getUint32(80, true);
      if (n > 0 && 84 + n * 50 === buf.byteLength) {
        var tris = new Float32Array(n * 9);
        for (var i = 0; i < n; i++) {
          var o = 84 + i * 50 + 12;
          for (var k = 0; k < 9; k++) tris[i * 9 + k] = dv.getFloat32(o + k * 4, true);
        }
        return { tris: tris, count: n };
      }
    }
    var txt = new TextDecoder('utf-8').decode(buf);
    var re = /vertex\s+(-?[\d.eE+\-]+)\s+(-?[\d.eE+\-]+)\s+(-?[\d.eE+\-]+)/g;
    var vals = [], m;
    while ((m = re.exec(txt))) vals.push(+m[1], +m[2], +m[3]);
    var nt = Math.floor(vals.length / 9);
    if (!nt) throw new Error('这个 STL 里没有三角形（二进制和 ASCII 都没解析出来）');
    return { tris: new Float32Array(vals.slice(0, nt * 9)), count: nt };
  }

  function parseOBJ(txt) {
    var verts = [], tris = [];
    var lines = txt.split(/\r?\n/);
    for (var li = 0; li < lines.length; li++) {
      var t = lines[li].trim().split(/\s+/);
      if (t[0] === 'v' && t.length >= 4) {
        verts.push(+t[1], +t[2], +t[3]);
      } else if (t[0] === 'f' && t.length >= 4) {
        var idx = [];
        for (var i = 1; i < t.length; i++) {
          var p = t[i].split('/')[0];
          if (!p) continue;
          var n = parseInt(p, 10);
          if (!isFinite(n)) continue;
          idx.push(n < 0 ? verts.length / 3 + n : n - 1);
        }
        for (var k = 1; k + 1 < idx.length; k++) {
          tris.push(idx[0], idx[k], idx[k + 1]);
        }
      }
    }
    if (!tris.length) throw new Error('这个 OBJ 里没有面（只支持 v + f 的三角/多边形网格）');
    var out = new Float32Array(tris.length * 3);
    for (var j = 0; j < tris.length; j++) {
      var b = tris[j] * 3;
      out[j * 3] = verts[b]; out[j * 3 + 1] = verts[b + 1]; out[j * 3 + 2] = verts[b + 2];
    }
    return { tris: out, count: tris.length / 3 };
  }

  function project(tris, axis) {
    var out = new Float32Array(tris.length / 3 * 2);
    for (var i = 0, j = 0; i < tris.length; i += 3, j += 2) {
      var x = tris[i], y = tris[i + 1], z = tris[i + 2];
      if (axis === 'y') { out[j] = x; out[j + 1] = z; }
      else if (axis === 'x') { out[j] = y; out[j + 1] = z; }
      else { out[j] = x; out[j + 1] = y; }
    }
    return out;
  }

  function makeCanvas(w, h) {
    var c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }

  /* 把透明底的轮廓图裁到内容边界，返回紧贴内容的画布。 */
  function cropAlpha(cv) {
    var g = cv.getContext('2d');
    var w = cv.width, h = cv.height;
    var d = g.getImageData(0, 0, w, h).data;
    var x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        if (d[(y * w + x) * 4 + 3] > 8) {
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
    }
    if (x1 < 0) throw new Error('投影后没有留下任何像素，换个投影轴或调整反相再试');
    var out = makeCanvas(x1 - x0 + 1, y1 - y0 + 1);
    out.getContext('2d').drawImage(cv, -x0, -y0);
    return { canvas: out, w: out.width, h: out.height };
  }

  function renderMesh(tris, axis) {
    var p = project(tris, axis);
    var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
    for (var i = 0; i < p.length; i += 2) {
      if (p[i] < mnx) mnx = p[i];
      if (p[i] > mxx) mxx = p[i];
      if (p[i + 1] < mny) mny = p[i + 1];
      if (p[i + 1] > mxy) mxy = p[i + 1];
    }
    var ext = Math.max(mxx - mnx, mxy - mny);
    if (!(ext > 1e-9)) throw new Error('模型在投影平面上退化成一条线，换个投影轴');
    var k = RES * (1 - 2 * PAD) / ext;
    var cv = makeCanvas(RES, RES);
    var g = cv.getContext('2d');
    g.fillStyle = '#fff';
    var path = new Path2D();
    for (var t = 0; t < p.length; t += 6) {
      var ax = (p[t] - mnx) * k + RES * PAD;
      var ay = RES - ((p[t + 1] - mny) * k + RES * PAD);
      var bx = (p[t + 2] - mnx) * k + RES * PAD;
      var by = RES - ((p[t + 3] - mny) * k + RES * PAD);
      var cx = (p[t + 4] - mnx) * k + RES * PAD;
      var cy = RES - ((p[t + 5] - mny) * k + RES * PAD);
      // nonzero 填充下正反绕向会互相抵消，先统一成同一绕向再并集
      if ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay) < 0) {
        var tx = bx, ty = by;
        bx = cx; by = cy; cx = tx; cy = ty;
      }
      path.moveTo(ax, ay); path.lineTo(bx, by); path.lineTo(cx, cy); path.closePath();
    }
    g.fill(path);
    return cropAlpha(cv);
  }

  function renderImage(img, invert) {
    var iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
    if (!iw || !ih) throw new Error('图片没有有效尺寸');
    var s = RES * (1 - 2 * PAD) / Math.max(iw, ih);
    var w = Math.max(1, Math.round(iw * s)), h = Math.max(1, Math.round(ih * s));
    var src = makeCanvas(w, h);
    var sg = src.getContext('2d');
    sg.drawImage(img, 0, 0, w, h);
    var d = sg.getImageData(0, 0, w, h).data;
    var hasAlpha = false;
    for (var i = 3; i < d.length; i += 4) if (d[i] < 250) { hasAlpha = true; break; }
    /* 有透明通道时把透明当背景；否则按亮度，默认深色是物体，invert 用来反过来。 */
    var out = makeCanvas(w, h);
    var og = out.getContext('2d');
    var id = og.createImageData(w, h);
    var o = id.data;
    for (var k = 0, p = 0; k < w * h; k++, p += 4) {
      var obj = hasAlpha
        ? d[p + 3] > 127
        : (0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2]) < 128;
      if (invert) obj = !obj;
      if (obj) { o[p] = 255; o[p + 1] = 255; o[p + 2] = 255; o[p + 3] = 255; }
    }
    og.putImageData(id, 0, 0);
    return cropAlpha(out);
  }

  function readFile(file, how) {
    return new Promise(function (ok, fail) {
      var r = new FileReader();
      r.onerror = function () { fail(new Error('读取文件失败：' + file.name)); };
      r.onload = function () { ok(r.result); };
      if (how === 'text') r.readAsText(file);
      else r.readAsArrayBuffer(file);
    });
  }

  function loadImageFile(file) {
    return new Promise(function (ok, fail) {
      var url = URL.createObjectURL(file);
      var img = new Image();
      img.onload = function () { URL.revokeObjectURL(url); ok(img); };
      img.onerror = function () { URL.revokeObjectURL(url); fail(new Error('图片解码失败：' + file.name)); };
      img.src = url;
    });
  }

  function load(file) {
    var name = file.name || '模型';
    var ext = name.split('.').pop().toLowerCase();
    if (ext === 'stl') {
      return readFile(file, 'buffer').then(function (buf) {
        var r = parseSTL(buf);
        return { kind: 'mesh', name: name, tris: r.tris, count: r.count };
      });
    }
    if (ext === 'obj') {
      return readFile(file, 'text').then(function (txt) {
        var r = parseOBJ(txt);
        return { kind: 'mesh', name: name, tris: r.tris, count: r.count };
      });
    }
    return loadImageFile(file).then(function (img) {
      return { kind: 'image', name: name, img: img };
    });
  }

  function render(src, opt) {
    if (!src) throw new Error('还没有导入模型');
    if (src.kind === 'mesh') return renderMesh(src.tris, (opt && opt.axis) || 'z');
    return renderImage(src.img, !!(opt && opt.invert));
  }

  global.ModelMask = { load: load, render: render, parseSTL: parseSTL, parseOBJ: parseOBJ };
})(typeof window !== 'undefined' ? window : globalThis);