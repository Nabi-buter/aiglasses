// Minimal ZIP reader for training data: lists entries from the central directory
// and reads them lazily (Blob.slice), so a large ZIP is never loaded whole.
// Supports STORE and DEFLATE (via DecompressionStream). No ZIP64, no encryption.

const SIG_EOCD = 0x06054b50, SIG_CEN = 0x02014b50, SIG_LOC = 0x04034b50;

function decodeName(bytes, utf8) {
  if (utf8) return new TextDecoder("utf-8").decode(bytes);
  try { return new TextDecoder("utf-8", { fatal: true }).decode(bytes); } catch {}
  try { return new TextDecoder("euc-kr").decode(bytes); } catch {}   // Windows Explorer ZIPs on Korean systems
  return new TextDecoder("latin1").decode(bytes);
}

/** -> [{name, size, method, read: () => Promise<Uint8Array>}] (directories skipped) */
export async function listZip(blob) {
  const tailLen = Math.min(blob.size, 65557);
  const tail = new DataView(await blob.slice(blob.size - tailLen).arrayBuffer());
  let eocd = -1;
  for (let i = tail.byteLength - 22; i >= 0; i--) if (tail.getUint32(i, true) === SIG_EOCD) { eocd = i; break; }
  if (eocd < 0) throw new Error("ZIP 파일이 아니거나 손상되었습니다.");
  const count = tail.getUint16(eocd + 10, true);
  const cdSize = tail.getUint32(eocd + 12, true), cdOff = tail.getUint32(eocd + 16, true);
  if (cdOff === 0xffffffff || count === 0xffff) throw new Error("4GB가 넘는 ZIP(ZIP64)은 지원하지 않습니다. 여러 개로 나눠 주세요.");
  const cd = new DataView(await blob.slice(cdOff, cdOff + cdSize).arrayBuffer());
  const out = [];
  for (let p = 0, i = 0; i < count && p + 46 <= cd.byteLength; i++) {
    if (cd.getUint32(p, true) !== SIG_CEN) throw new Error("ZIP 목록이 손상되었습니다.");
    const flags = cd.getUint16(p + 8, true), method = cd.getUint16(p + 10, true);
    const csize = cd.getUint32(p + 20, true), size = cd.getUint32(p + 24, true);
    const nl = cd.getUint16(p + 28, true), xl = cd.getUint16(p + 30, true), cl = cd.getUint16(p + 32, true);
    const loc = cd.getUint32(p + 42, true);
    const name = decodeName(new Uint8Array(cd.buffer, cd.byteOffset + p + 46, nl), flags & 0x800).replace(/\\/g, "/");
    p += 46 + nl + xl + cl;
    if (name.endsWith("/") || flags & 1) continue;                    // folder or encrypted
    out.push({
      name, size, method,
      async read() {
        const lh = new DataView(await blob.slice(loc, loc + 30).arrayBuffer());
        if (lh.getUint32(0, true) !== SIG_LOC) throw new Error("ZIP 항목이 손상되었습니다: " + name);
        const start = loc + 30 + lh.getUint16(26, true) + lh.getUint16(28, true);
        const raw = blob.slice(start, start + csize);
        if (method === 0) return new Uint8Array(await raw.arrayBuffer());
        if (method === 8) return new Uint8Array(await new Response(raw.stream().pipeThrough(new DecompressionStream("deflate-raw"))).arrayBuffer());
        throw new Error(`지원하지 않는 압축 방식(${method}): ${name}`);
      },
    });
  }
  return out;
}
