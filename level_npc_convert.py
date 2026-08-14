#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
level_npc_convert.py  —  Converte NPC story-mode -> MPLAYER con texture isolata
================================================================================
Prende l'intera cartella di un livello (tutti i MODEL.HMD degli NPC),
isola i pixel texture appartenenti ESCLUSIVAMENTE all'NPC scelto rasterizzando
i triangoli UV di tutti i modelli, e produce una texture pulita + OBJ/HMD/TIM.

Funzionamento:
  1. Scansiona la cartella e trova tutti i MODEL.HMD
  2. L'utente sceglie quale NPC convertire
  3. Per ogni pagina texture condivisa (TP14, TP30, ecc.):
     - Rasterizza i triangoli UV di TUTTI gli NPC -> mappa di copertura
     - I pixel coperti solo dal target = suoi pixel esclusivi
     - Gli altri NPC "escludono" le loro zone dalla texture del target
  4. Usa TP15 (Gabe, non condivisa) come da solo
  5. Costruisce combined texture con embed nelle righe libere di TP15
  6. Produce OBJ + MTL + HMD + TIM pronti per MPLAYER

Uso:
    python level_npc_convert.py <cartella_livello>
        --body body.tim --face face.tim --gabe gabe.tim
        [--target sf2|sf3]

    Con selezione interattiva se --npc non e' specificato.
    Con --npc <nome> per uso non-interattivo.
"""

import struct, os, sys, argparse
import numpy as np
from PIL import Image, ImageDraw

# ── Costanti MPLAYER ──────────────────────────────────────────────────────────
MPLAYER_TSB      = 143   # page 15, U=[0,127]  — standard 128x256
MPLAYER_TSB_WIDE = 136   # page 8,  U=[0,255]  — 256x256, PIX_ORG_X=512 coincide
MPLAYER_CBA      = 32688

PAGE_LABELS = {
    # Palette (ZCLUT.BIN): CLUT 10 per tutti tranne Gabe (CLUT 30/penultima)
    (960,   0): 'gabe',       # TP15 — Gabe, CLUT 30
    (896,   0): 'body',       # TP14 — corpo NPC, CLUT 10
    (896, 256): 'face',       # TP30 — faccia NPC, CLUT 10
    (832, 256): 'face_alt',   # TP29 — multi-faccia SF3, CLUT 10  ← --page 832,256=...
    (768, 256): 'face_alt2',  # TP28 — CLUT 10, PARK2/hans
    (384,   0): 'special',    # WRECK/archer
}

T_EFF = {
    'Head':(0,63,-1),'Chest':(0,13,0),'Hips':(0,0,0),
    'LeftUpArm':(18,52,-2),'LeftLowArm':(18,21,-2),'LeftHand':(18,-8,-2),
    'LeftUpLeg':(9,4,1),'LeftLowLeg':(10,-44,-2),'LeftFoot':(11,-96,-2),
    'RightUpArm':(-18,52,-1),'RightLowArm':(-18,18,-1),'RightHand':(-18,-10,-1),
    'RightUpLeg':(-10,3,0),'RightLowLeg':(-13,-44,-2),'RightFoot':(-14,-95,-2),
}

# ── Utility ───────────────────────────────────────────────────────────────────
def is_valid_name(nb):
    nul=nb.find(0)
    if nul<=0 or nul>15: return False
    try: s=nb[:nul].decode('ascii')
    except: return False
    return all(c.isalnum() or c in '_- ' for c in s)

def decode_tsb(tsb):
    return ((tsb&0xF)*64, ((tsb>>4)&1)*256)

def load_image(path):
    """Carica PNG o TIM 8bpp, ritorna PIL Image RGB."""
    with open(path,'rb') as f: raw=f.read()
    if len(raw)>8 and raw[0]==0x10 and raw[4]==0x09:
        cl=struct.unpack_from('<I',raw,8)[0]
        pal=[]
        for i in range(256):
            v=struct.unpack_from('<H',raw,8+12+i*2)[0] if 8+12+i*2+2<=len(raw) else 0
            pal.append((0,0,0) if v==0 else ((v&31)<<3,((v>>5)&31)<<3,((v>>10)&31)<<3))
        px=raw[8+cl+12:]
        img=Image.new('RGB',(128,256)); pix=img.load()
        for y in range(256):
            for x in range(128):
                idx=y*128+x; pix[x,y]=pal[px[idx] if idx<len(px) else 0]
        return img
    return Image.open(path).convert('RGB')

# ── Parser MODEL.HMD ──────────────────────────────────────────────────────────
def parse_hmd(blob):
    """Ritorna lista di poligoni con UV e pagina, e lista di bone con vertici."""
    if struct.unpack_from('<I',blob,0)[0]!=0x48000000:
        return None, None
    pc=struct.unpack_from('<I',blob,8)[0]//4
    vo=min(struct.unpack_from('<I',blob,0x14)[0],len(blob))
    secs=[]
    for i in range(0x24+pc*16,min(vo,len(blob)-8),4):
        if struct.unpack_from('<I',blob,i+4)[0]==pc and is_valid_name(blob[i+0x28:i+0x28+16]):
            secs.append(i)
    # Auto-detect vref divisor
    total_nV=sum(struct.unpack_from('<H',blob,s+0x28+12)[0] for s in secs)
    bad8=sum(1 for pi in range(pc) for off in [0x24+pi*16+10,0x24+pi*16+12,0x24+pi*16+14]
             if 0x24+pi*16+16<=len(blob) and struct.unpack_from('<H',blob,off)[0]//8>=total_nV)
    div=12 if bad8>0 else 8

    polys=[]
    for pi in range(pc):
        o=0x24+pi*16
        if o+16>len(blob): break
        tsb=struct.unpack_from('<H',blob,o+6)[0]
        page=decode_tsb(tsb)
        u0,v0=blob[o],blob[o+1]; u1,v1=blob[o+4],blob[o+5]; u2,v2=blob[o+8],blob[o+9]
        r0=struct.unpack_from('<H',blob,o+10)[0]//div
        r1=struct.unpack_from('<H',blob,o+12)[0]//div
        r2=struct.unpack_from('<H',blob,o+14)[0]//div
        polys.append({'page':page,'uvs':[(u0,v0),(u1,v1),(u2,v2)],'refs':[r0,r1,r2]})

    bones=[]
    for s in secs:
        tx,ty=struct.unpack_from('<2h',blob,s+0x24)
        ns=s+0x28; ne=blob[ns:ns+20].find(0)
        name=blob[ns:ns+ne].decode('ascii','replace').strip() if ne>=0 else '?'
        co=ns+12; nV=struct.unpack_from('<H',blob,co)[0]; vs=co+16
        verts=[]
        for vi in range(nV):
            if vs+vi*8+6>len(blob): break
            x,y,z,_=struct.unpack_from('<4h',blob,vs+vi*8)
            verts.append((x,y,z))
        bones.append({'name':name,'tx':tx,'ty':ty,'nV':nV,'verts':verts})
    return polys, bones, div

# ── Rasterizzazione triangoli UV ──────────────────────────────────────────────
def rasterize_uv_triangles(polys, page, mask):
    """
    Riempie la maschera numpy (128x256) con True per ogni pixel coperto
    da almeno un triangolo UV del modello per la pagina specificata.
    Usa scanline rasterization per coprire i pixel interni ai triangoli.
    """
    for p in polys:
        if p['page']!=page: continue
        (u0,v0),(u1,v1),(u2,v2)=p['uvs']
        # Bounding box del triangolo
        min_v=max(0,min(v0,v1,v2)); max_v=min(255,max(v0,v1,v2))
        min_u=max(0,min(u0,u1,u2)); max_u=min(127,max(u0,u1,u2))
        # Scanline: per ogni riga V nel bounding box trova U range
        for v in range(min_v, max_v+1):
            u_hits=[]
            for (ua,va),(ub,vb) in [((u0,v0),(u1,v1)),((u1,v1),(u2,v2)),((u2,v2),(u0,v0))]:
                if (va<=v<vb) or (vb<=v<va):
                    if vb!=va:
                        u_hit=ua+(ub-ua)*(v-va)/(vb-va)
                        u_hits.append(u_hit)
            if len(u_hits)>=2:
                ul,ur=int(min(u_hits)),int(max(u_hits))+1
                mask[v, max(0,ul):min(128,ur)]=True
            # Marca sempre i vertici stessi
        mask[v0,u0]=True; mask[v1,u1]=True; mask[v2,u2]=True

# ── Scansione cartella livello ────────────────────────────────────────────────
def scan_level_folder(folder):
    """Trova tutti i MODEL.HMD nella cartella (ricorsivo)."""
    found=[]
    for root,dirs,files in os.walk(folder):
        if 'MODEL.HMD' in files:
            path=os.path.join(root,'MODEL.HMD')
            relname=os.path.relpath(root,folder)
            found.append((relname, path))
    return sorted(found)

# ── Costruzione combined texture con embed esclusivo ─────────────────────────
def build_exclusive_texture(target_polys, all_polys_by_npc, page_textures,
                            verbose=True, wide=False, CW_override=None, repack=False):
    """
    Sceglie automaticamente la strategia di texture building:
    - repack=True: island packing PURO — ogni pagina (TP15 inclusa) contribuisce
      solo le isole UV realmente usate, impacchettate per risoluzione massima.
      Scarta le parti non usate (es. pixel di Gabe che l'NPC non tocca).
    - Con Gabe (TP15 presente): TP15 come base + embed secondarie
    - Senza Gabe: island packing da zero
    Se wide=True usa canvas 256x256 invece di 128x256.
    """
    CW = CW_override if CW_override else (256 if wide else 128)
    CH = 256
    if repack:
        if verbose:
            print(f"  Strategia: island packing PURO (repack) {'[256x256]' if wide else '[128x256]'}")
        return _build_island_packed(target_polys, all_polys_by_npc, page_textures, verbose, CW, CH)
    has_gabe = any(p['page'] == (960, 0) for p in target_polys) and (960, 0) in page_textures
    if has_gabe:
        if verbose:
            print(f"  Strategia: TP15 base + embed secondarie {'[256x256]' if wide else '[128x256]'}")
        return _build_gabe_based(target_polys, all_polys_by_npc, page_textures, verbose, CW, CH)
    else:
        if verbose:
            print(f"  Strategia: island packing da zero {'[256x256]' if wide else '[128x256]'}")
        return _build_island_packed(target_polys, all_polys_by_npc, page_textures, verbose, CW, CH)


def _build_gabe_based(target_polys, all_polys_by_npc, page_textures, verbose=True, CW=128, CH=256):
    """
    Strategia per NPC con Gabe (SF1/SF2 story-mode):
    TP15 (suit) come base pixel-perfect, pagine secondarie (face, body)
    embedded ciascuna nel blocco libero piu' grande disponibile.
    UV del suit invariate. UV secondarie rimappate linearmente (un blocco
    per pagina = nessuna interpolazione spuria sul PS1).
    """
    from scipy.ndimage import binary_dilation
    import numpy as np

    primary = (960, 0)

    def rast(polys, page):
        mask = np.zeros((CH, CW), bool)
        for p in polys:
            if p['page'] != page: continue
            (u0,v0),(u1,v1),(u2,v2) = p['uvs']
            for v in range(max(0,min(v0,v1,v2)), min(CH-1,max(v0,v1,v2))+1):
                hits=[]
                for (ua,va),(ub,vb) in [((u0,v0),(u1,v1)),((u1,v1),(u2,v2)),((u2,v2),(u0,v0))]:
                    if (va<=v<vb) or (vb<=v<va):
                        hits.append(ua+(ub-ua)*(v-va)/(vb-va))
                if len(hits)>=2:
                    mask[v, max(0,int(min(hits))):min(CW,int(max(hits))+1)] = True
            if 0<=v0<CH and 0<=u0<CW: mask[v0,u0]=True
            if 0<=v1<CH and 0<=u1<CW: mask[v1,u1]=True
            if 0<=v2<CH and 0<=u2<CW: mask[v2,u2]=True
        return mask

    # Base: canvas CW×CH (dark) con TP15 nella metà sinistra (sempre 128px wide)
    TP15_W = 128  # TP15 e' sempre 128 pixel wide
    base_img = load_image(page_textures[primary]).convert('RGB')
    combined_arr = np.full((CH, CW, 3), 15, dtype=np.uint8)  # sfondo scuro
    # Incolla TP15 nella colonna 0..TP15_W-1
    combined_arr[:, :TP15_W, :] = np.array(base_img)[:CH, :TP15_W, :]

    # Righe libere nel TP15 (non usate dal suit) — cercate solo su colonne 0..TP15_W
    suit_mask_128 = np.zeros((CH, TP15_W), bool)
    for p in target_polys:
        if p['page'] != primary: continue
        (u0,v0),(u1,v1),(u2,v2) = p['uvs']
        for v in range(max(0,min(v0,v1,v2)), min(CH-1,max(v0,v1,v2))+1):
            hits=[]
            for (ua,va),(ub,vb) in [((u0,v0),(u1,v1)),((u1,v1),(u2,v2)),((u2,v2),(u0,v0))]:
                if (va<=v<vb) or (vb<=v<va):
                    hits.append(ua+(ub-ua)*(v-va)/(vb-va))
            if len(hits)>=2:
                suit_mask_128[v, max(0,int(min(hits))):min(TP15_W,int(max(hits))+1)] = True
        suit_mask_128[min(v0,CH-1),min(u0,TP15_W-1)]=True
        suit_mask_128[min(v1,CH-1),min(u1,TP15_W-1)]=True
        suit_mask_128[min(v2,CH-1),min(u2,TP15_W-1)]=True

    used_rows = set(int(y) for y in np.where(suit_mask_128.any(axis=1))[0])
    free_rows = [y for y in range(CH) if y not in used_rows]
    groups = []; s = None
    for y in sorted(free_rows):
        if s is None: s=y; p2=y
        elif y==p2+1: p2=y
        else: groups.append((s,p2)); s=y; p2=y
    if s is not None: groups.append((s,p2))
    groups.sort(key=lambda g: g[1]-g[0], reverse=True)

    uv_remap = {}
    FACE_PAGES = {(896,256),(896,224),(832,256),(768,256)}
    sec_pages = sorted(
        [pg for pg in set(p['page'] for p in target_polys)
         if pg != primary and pg in page_textures],
        key=lambda pg: (0 if pg in FACE_PAGES else 1,
                        -sum(1 for p in target_polys if p['page']==pg)))

    if CW > 128:
        # ── WIDE 256x256: layout colonne dinamiche ───────────────────────────
        # U=0-127  : TP15 suit (pixel-perfect, gia' piazzato sopra)
        # U=128-255: N pagine secondarie, ognuna in una colonna da 128/N px
        # (prima erano solo 2 colonne fisse a 128/192 → 3+ pagine si sovrascrivevano)
        sec_present = [pg for pg in sec_pages if pg in page_textures]
        n_sec = len(sec_present)
        if n_sec == 0:
            n_sec = 1  # evita div/0
        AVAIL_W = 128  # spazio disponibile per le secondarie (U=128..255)
        COL_W = max(1, AVAIL_W // n_sec)
        # Assegna a ogni pagina secondaria la sua colonna
        col_of = {}
        for i, pg in enumerate(sec_present):
            col_of[pg] = 128 + i * COL_W
        if verbose and n_sec > 2:
            print(f"    Layout: {n_sec} pagine secondarie → colonne da {COL_W}px "
                  f"(fix multi-pagina)")

        for sec_page in sec_pages:
            if sec_page not in page_textures: continue
            cs = col_of[sec_page]
            src_arr = np.array(load_image(page_textures[sec_page]))

            # UV bake: rasterizza solo i triangoli dell'NPC, zero contaminazione
            baked = np.zeros((CH, 128, 3), dtype=np.uint8)
            covered = np.zeros((CH, 128), bool)
            for p in target_polys:
                if p['page'] != sec_page: continue
                (u0,v0),(u1,v1),(u2,v2) = p['uvs']
                for sv in range(max(0,min(v0,v1,v2)), min(CH-1,max(v0,v1,v2))+1):
                    hits=[]
                    for (ua,va),(ub,vb) in [((u0,v0),(u1,v1)),((u1,v1),(u2,v2)),((u2,v2),(u0,v0))]:
                        if (va<=sv<vb) or (vb<=sv<va):
                            hits.append(ua+(ub-ua)*(sv-va)/(vb-va))
                    if len(hits)>=2:
                        for su in range(max(0,int(min(hits))), min(127,int(max(hits)))+1):
                            if su<src_arr.shape[1] and sv<src_arr.shape[0]:
                                baked[sv,su]=src_arr[sv,su]; covered[sv,su]=True
                for u,v in [(u0,v0),(u1,v1),(u2,v2)]:
                    if 0<=u<128 and 0<=v<CH and u<src_arr.shape[1] and v<src_arr.shape[0]:
                        baked[v,u]=src_arr[v,u]; covered[v,u]=True

            # Nearest-fill aree scoperte
            if covered.any():
                from scipy.ndimage import distance_transform_edt
                _, idx = distance_transform_edt(~covered, return_indices=True)
                baked[~covered] = baked[idx[0][~covered], idx[1][~covered]]

            # Scala a COL_W e piazza nel canvas
            scaled = np.array(Image.fromarray(baked).resize((COL_W, CH), Image.LANCZOS))
            combined_arr[:, cs:cs+COL_W, :] = scaled
            # UV remap: u_new = cs + round(u * COL_W/128)
            uv_remap[sec_page] = {}
            for p in target_polys:
                if p['page'] != sec_page: continue
                for u, v in p['uvs']:
                    if (u, v) not in uv_remap[sec_page]:
                        nu = cs + min(COL_W-1, round(u * COL_W / 128))
                        uv_remap[sec_page][(u,v)] = (nu, min(CH-1, v))
            label = PAGE_LABELS.get(sec_page, str(sec_page))
            if verbose:
                print(f"    {label}: U=[{cs},{cs+COL_W-1}] V=[0,255] "
                      f"(128→{COL_W}px wide, full height)")

    else:
        # ── STANDARD 128x256: distribuzione proporzionale righe libere ────────
        total_free = sum(ge-gs+1 for gs,ge in groups)
        page_src_h = {}
        for pg in sec_pages:
            if pg not in page_textures: continue
            vs = [v for p in target_polys if p['page']==pg for u,v in p['uvs']]
            if vs: page_src_h[pg] = (min(vs), max(vs), max(vs)-min(vs)+1)

        total_src = sum(v[2] for v in page_src_h.values())
        cursor = groups[0][0] if groups else 0
        remaining = total_free

        for i, sec_page in enumerate(sec_pages):
            if sec_page not in page_src_h: continue
            uv_v_min, uv_v_max, src_h = page_src_h[sec_page]

            if remaining <= 0:
                if verbose: print(f"    {PAGE_LABELS.get(sec_page,str(sec_page))}: nessun spazio libero")
                continue

            # Alloca righe proporzionalmente
            pages_remaining = sum(1 for pg in sec_pages[i:] if pg in page_src_h)
            if pages_remaining <= 1:
                dst_h = remaining
            else:
                dst_h = max(1, round(total_free * src_h / max(1, total_src)))
                dst_h = min(dst_h, remaining - (pages_remaining-1))
            dst_s = cursor
            dst_e = min(CH-1, cursor + dst_h - 1)
            dst_h = dst_e - dst_s + 1
            cursor += dst_h
            remaining -= dst_h

            src_arr = np.array(load_image(page_textures[sec_page]))
            src_w = src_arr.shape[1]
            target_mask = rast(target_polys, sec_page)
            embed_mask = binary_dilation(target_mask, iterations=6)
            src_chunk = src_arr[uv_v_min:uv_v_max+1, :, :]
            clean = src_chunk.copy()
            for dy in range(src_h):
                row_mask = embed_mask[min(uv_v_min+dy, CH-1)][:src_w]
                not_cov = ~row_mask
                if not_cov.any() and row_mask.any():
                    cov_cols = np.where(row_mask)[0]
                    for cx in np.where(not_cov)[0]:
                        clean[dy, min(cx,src_w-1)] = clean[dy, cov_cols[np.argmin(np.abs(cov_cols-cx))]]
            if src_h != dst_h:
                scaled = np.array(Image.fromarray(clean).resize((src_w, dst_h), Image.LANCZOS))
            else:
                scaled = clean
            combined_arr[dst_s:dst_e+1, :src_w, :] = scaled
            uv_remap[sec_page] = {}
            for p in target_polys:
                if p['page'] != sec_page: continue
                for u, v in p['uvs']:
                    if (u, v) not in uv_remap[sec_page]:
                        if uv_v_max > uv_v_min:
                            t = (v - uv_v_min) / (uv_v_max - uv_v_min)
                            nv = dst_s + round(t * (dst_e - dst_s))
                        else:
                            nv = dst_s
                        uv_remap[sec_page][(u,v)] = (u, max(dst_s, min(dst_e, nv)))
            if verbose:
                ratio = 100*dst_h//max(1,src_h)
                print(f"    {PAGE_LABELS.get(sec_page,str(sec_page))}: "
                      f"V=[{uv_v_min},{uv_v_max}] -> V=[{dst_s},{dst_e}] ({ratio}%)")

    # Post-fill
    for dy in range(CH-1):
        row = combined_arr[dy]
        is_w = (row[:,0]>230)&(row[:,1]>230)&(row[:,2]>230)
        if is_w.any():
            combined_arr[dy, is_w] = combined_arr[dy-1 if dy>0 else dy+1, is_w]

    return Image.fromarray(combined_arr), uv_remap

    # ── Pre-calcola allocazione proporzionale in standard mode ──────────────
    # Raccoglie tutte le pagine secondarie con le loro sorgenti
    # e distribuisce lo spazio libero disponibile proporzionalmente
    if CW <= 128:
        # Spazio totale disponibile
        total_free_rows = sum(ge-gs+1 for gs,ge in groups)
        
        # Calcola src_h per ogni pagina secondaria
        page_src_info = {}
        for pg in sec_pages:
            if pg not in page_textures: continue
            vs = [v for p in target_polys if p['page']==pg for u,v in p['uvs']]
            if vs:
                page_src_info[pg] = (min(vs), max(vs), max(vs)-min(vs)+1)
        
        total_src_h = sum(info[2] for info in page_src_info.values())
        
        # Alloca righe proporzionalmente (min 1 riga per pagina presente)
        page_alloc = {}
        if total_src_h > 0 and total_free_rows > 0:
            cursor = groups[0][0] if groups else 0  # inizia dal primo blocco libero
            remaining = total_free_rows
            pages_left = list(page_src_info.keys())
            for i, pg in enumerate(pages_left):
                src_h = page_src_info[pg][2]
                if i == len(pages_left)-1:
                    h = remaining  # ultima pagina prende il resto
                else:
                    h = max(1, round(total_free_rows * src_h / total_src_h))
                    h = min(h, remaining - (len(pages_left)-i-1))
                page_alloc[pg] = (cursor, cursor+h-1, h)
                cursor += h
                remaining -= h

    # Post-fill
    for dy in range(CH-1):
        row = combined_arr[dy]
        is_w = (row[:,0]>230)&(row[:,1]>230)&(row[:,2]>230)
        if is_w.any():
            combined_arr[dy, is_w] = combined_arr[dy-1 if dy>0 else dy+1, is_w]

    return Image.fromarray(combined_arr), uv_remap


def _build_island_packed(target_polys, all_polys_by_npc, page_textures, verbose=True, CW=128, CH=256):
    """
    Strategia per NPC senza Gabe (SF3 story-mode).
    In wide mode (CW=256): layout semplice TP14 sinistra (U=0-127), TP30 destra (U=128-255).
    In standard mode (CW=128): island packing da zero.
    """
    from scipy.ndimage import binary_dilation, label as scipy_label
    import numpy as np

    FACE_PAGES = {(896,256),(896,224),(832,256),(768,256)}
    BODY_PAGES = {(896,0),(832,0)}

    # ── Wide mode: UV bake a N colonne dinamiche ──────────────────────────────
    if CW > 128:
        canvas = np.full((CH, CW, 3), 20, dtype=np.uint8)
        uv_remap = {}

        pages_used = [pg for pg in sorted(set(p['page'] for p in target_polys),
                      key=lambda pg: -sum(1 for p in target_polys if p['page']==pg))
                      if pg in page_textures]

        body_pgs = [pg for pg in pages_used if pg in BODY_PAGES]
        face_pgs = [pg for pg in pages_used if pg not in BODY_PAGES]

        def assign_cols(pages, start, total_w):
            if not pages: return {}
            w = total_w // len(pages)
            return {pg: (start + i*w, w) for i,pg in enumerate(pages)}

        col_all = {**assign_cols(body_pgs, 0, 128),
                   **assign_cols(face_pgs, 128, 128)}

        from scipy.ndimage import distance_transform_edt

        for pg, (col_start, col_w) in col_all.items():
            src = np.array(Image.open(page_textures[pg]).convert('RGB'))
            src_w = min(128, src.shape[1])

            baked   = np.zeros((CH, src_w, 3), dtype=np.uint8)
            covered = np.zeros((CH, src_w), bool)

            for p in target_polys:
                if p['page'] != pg: continue
                (u0,v0),(u1,v1),(u2,v2) = p['uvs']
                for sv in range(max(0,min(v0,v1,v2)), min(CH-1,max(v0,v1,v2))+1):
                    hits=[]
                    for (ua,va),(ub,vb) in [((u0,v0),(u1,v1)),((u1,v1),(u2,v2)),((u2,v2),(u0,v0))]:
                        if (va<=sv<vb) or (vb<=sv<va):
                            hits.append(ua+(ub-ua)*(sv-va)/(vb-va))
                    if len(hits)>=2:
                        for su in range(max(0,int(min(hits))), min(src_w-1,int(max(hits)))+1):
                            if sv<src.shape[0]:
                                baked[sv,su]=src[sv,su]; covered[sv,su]=True
                for u,v in [(u0,v0),(u1,v1),(u2,v2)]:
                    su,sv=min(u,src_w-1),min(v,CH-1)
                    if 0<=sv<CH and 0<=su<src_w and sv<src.shape[0]:
                        baked[sv,su]=src[sv,su]; covered[sv,su]=True

            if covered.any():
                _, idx = distance_transform_edt(~covered, return_indices=True)
                baked[~covered] = baked[idx[0][~covered], idx[1][~covered]]

            # Scala a col_w e piazza
            scaled = np.array(Image.fromarray(baked).resize((col_w,CH),Image.LANCZOS))
            canvas[:, col_start:col_start+col_w, :] = scaled

            # UV remap: scala u → col_start + round(u*col_w/src_w)
            uv_remap[pg] = {}
            for p in target_polys:
                if p['page'] != pg: continue
                for u,v in p['uvs']:
                    if (u,v) not in uv_remap[pg]:
                        nu = col_start + min(col_w-1, round(u*col_w/src_w))
                        uv_remap[pg][(u,v)] = (nu, min(CH-1,v))

            label = PAGE_LABELS.get(pg, str(pg))
            if verbose:
                print(f"    {label}: U=[{col_start},{col_start+col_w-1}] "
                      f"({col_w}px, UV bake {covered.sum()}px)")

        return Image.fromarray(canvas), uv_remap

    # ── Standard mode: island packing ─────────────────────────────────────────

    def rast(polys, page):
        mask = np.zeros((CH, CW), bool)
        for p in polys:
            if p['page'] != page: continue
            (u0,v0),(u1,v1),(u2,v2) = p['uvs']
            for v in range(max(0,min(v0,v1,v2)), min(CH-1,max(v0,v1,v2))+1):
                hits = []
                for (ua,va),(ub,vb) in [((u0,v0),(u1,v1)),((u1,v1),(u2,v2)),((u2,v2),(u0,v0))]:
                    if (va<=v<vb) or (vb<=v<va):
                        hits.append(ua+(ub-ua)*(v-va)/(vb-va))
                if len(hits)>=2:
                    mask[v, max(0,int(min(hits))):min(CW,int(max(hits))+1)] = True
            if 0<=v0<CH and 0<=u0<CW: mask[v0,u0]=True
            if 0<=v1<CH and 0<=u1<CW: mask[v1,u1]=True
            if 0<=v2<CH and 0<=u2<CW: mask[v2,u2]=True
        return mask

    # ── 1. Trova islands per ogni pagina ──────────────────────────────────────
    pages_used = sorted(set(p['page'] for p in target_polys
                            if p['page'] in page_textures),
                        key=lambda pg: -sum(1 for p in target_polys if p['page']==pg))

    islands = []  # {page, src_bb, src_mask, comp_mask, w, h, px}
    for page in pages_used:
        target_mask = rast(target_polys, page)
        # Dilata 2px per bordi puliti
        dilated = binary_dilation(target_mask, iterations=2)

        labeled, n = scipy_label(dilated)
        for cid in range(1, n+1):
            cm = labeled == cid
            rows = np.where(cm.any(axis=1))[0]
            cols = np.where(cm.any(axis=0))[0]
            if len(rows) == 0: continue
            v1r,v2r = int(rows.min()), int(rows.max())
            u1c,u2c = int(cols.min()), int(cols.max())
            w, h = u2c-u1c+1, v2r-v1r+1
            islands.append({
                'page': page,
                'bb': (u1c,v1r,u2c,v2r),
                'src_mask': target_mask,
                'comp_mask': cm,
                'w': w, 'h': h,
                'px': int(cm.sum()),
                'scale': 1.0, 'scaled_h': h,
            })

    if not islands:
        if verbose: print("  Nessuna island trovata")
        return None, {}

    # ── 1b. Merge islands che condividono vertici dello stesso triangolo ──────
    # Senza questo, un triangolo con vertici in 2 islands diverse causa
    # interpolazione UV che passa per zone sbagliate (texture "a caso").
    # Soluzione: unisci le islands collegate da almeno un triangolo.
    changed = True
    while changed:
        changed = False
        for p in target_polys:
            page = p['page']
            # Trova quali islands contengono i vertici di questo triangolo
            isl_for_poly = set()
            for u, v in p['uvs']:
                for idx, isl in enumerate(islands):
                    if isl['page'] != page: continue
                    u1,v1,u2,v2 = isl['bb']
                    if u1<=u<=u2 and v1<=v<=v2:
                        isl_for_poly.add(idx); break
            if len(isl_for_poly) > 1:
                # Merge: unisci tutte in una sola island (bounding box unione)
                idxs = sorted(isl_for_poly)
                base = islands[idxs[0]]
                for i in idxs[1:]:
                    other = islands[i]
                    if other['page'] != base['page']: continue
                    u1 = min(base['bb'][0], other['bb'][0])
                    v1 = min(base['bb'][1], other['bb'][1])
                    u2 = max(base['bb'][2], other['bb'][2])
                    v2 = max(base['bb'][3], other['bb'][3])
                    base['bb'] = (u1,v1,u2,v2)
                    base['comp_mask'] = base['comp_mask'] | other['comp_mask']
                    base['w'] = u2-u1+1; base['h'] = v2-v1+1
                    base['px'] = int(base['comp_mask'].sum())
                # Rimuovi le islands mergiate (in ordine inverso)
                for i in sorted(idxs[1:], reverse=True):
                    islands.pop(i)
                changed = True
                break  # ricomincia il while con la lista aggiornata

    # Riordina dopo merge
    islands.sort(key=lambda i: i['w']*i['h'], reverse=True)

    if verbose:
        total_px = sum(i['px'] for i in islands)
        print(f"  UV islands: {len(islands)} su {len(pages_used)} pagine  "
              f"pixel totali={total_px}/{CW*CH} ({100*total_px//(CW*CH)}%)")

    # ── 2. Packing 2D (bin packing) ───────────────────────────────────────────
    canvas_used = np.zeros((CH, CW), bool)

    def try_place(w, h):
        for v in range(CH-h+1):
            for u in range(CW-w+1):
                if not canvas_used[v:v+h, u:u+w].any():
                    return u, v
        return None, None

    MARGIN = 1

    # Pre-scaling proattivo: se le isole full-width (w vicino a CW) hanno altezza
    # totale > CH, non entreranno mai impilate. Scala le più alte per far stare tutto.
    fullwidth = [i for i in islands if i['w'] >= CW * 0.75]
    fw_height = sum(i['h'] + MARGIN for i in fullwidth)
    if fw_height > CH and fullwidth:
        # Fattore di scala per far entrare tutte le full-width in altezza
        target_factor = CH / fw_height * 0.95
        for isl in fullwidth:
            new_h = max(8, int(isl['h'] * target_factor))
            new_w = max(8, int(isl['w'] * target_factor))
            isl['_prescale'] = (new_w, new_h)
        if verbose:
            print(f"    Pre-scaling {len(fullwidth)} isole full-width "
                  f"(altezza totale {fw_height} > {CH}) → fattore {target_factor:.0%}")

    placed = []
    pending = []

    for isl in islands:
        # Applica pre-scaling se presente
        if '_prescale' in isl:
            pw, ph = isl['_prescale']
            u, v = try_place(pw + MARGIN, ph + MARGIN)
            if u is not None:
                canvas_used[v:v+ph+MARGIN, u:u+pw+MARGIN] = True
                isl['dst_u'] = u; isl['dst_v'] = v
                isl['scaled_w'] = pw; isl['scaled_h'] = ph
                isl['scale'] = pw / isl['w']
                isl['offset_u'] = u - isl['bb'][0]
                isl['offset_v'] = v - isl['bb'][1]
                placed.append(isl)
                if verbose:
                    print(f"    Island {PAGE_LABELS.get(isl['page'],str(isl['page']))} "
                          f"{isl['w']}x{isl['h']} -> pre-scala -> {pw}x{ph}")
                continue
            else:
                pending.append(isl); continue
        mx = 0 if isl['w'] >= CW else MARGIN
        u, v = try_place(isl['w']+mx, isl['h']+MARGIN)
        if u is not None:
            canvas_used[v:v+isl['h']+MARGIN, u:u+isl['w']+mx] = True
            isl['dst_u'] = u; isl['dst_v'] = v
            isl['offset_u'] = u - isl['bb'][0]
            isl['offset_v'] = v - isl['bb'][1]
            placed.append(isl)
        else:
            pending.append(isl)

    # Islands che non entrano: prima scala verticale, poi scala 2D se serve
    for isl in pending:
        placed_ok = False
        # Fase 1: scala solo verticale (preserva risoluzione orizzontale)
        for scale in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
            sh = max(4, round(isl['h'] * scale))
            u, v = try_place(isl['w'], sh)
            if u is not None:
                canvas_used[v:v+sh, u:u+isl['w']] = True
                isl['dst_u'] = u; isl['dst_v'] = v
                isl['scale'] = scale; isl['scaled_h'] = sh
                isl['scaled_w'] = isl['w']
                isl['offset_u'] = u - isl['bb'][0]
                isl['offset_v'] = v - isl['bb'][1]
                placed.append(isl)
                placed_ok = True
                if verbose:
                    print(f"    Island {PAGE_LABELS.get(isl['page'],str(isl['page']))} "
                          f"{isl['w']}x{isl['h']} -> scala V {scale:.0%} -> {isl['w']}x{sh}")
                break
        # Fase 2: scala 2D (larghezza + altezza) per isole troppo larghe
        if not placed_ok:
            for scale in [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]:
                sw = max(4, round(isl['w'] * scale))
                sh = max(4, round(isl['h'] * scale))
                u, v = try_place(sw, sh)
                if u is not None:
                    canvas_used[v:v+sh, u:u+sw] = True
                    isl['dst_u'] = u; isl['dst_v'] = v
                    isl['scale'] = scale; isl['scaled_h'] = sh
                    isl['scaled_w'] = sw
                    isl['offset_u'] = u - isl['bb'][0]
                    isl['offset_v'] = v - isl['bb'][1]
                    placed.append(isl)
                    placed_ok = True
                    if verbose:
                        print(f"    Island {PAGE_LABELS.get(isl['page'],str(isl['page']))} "
                              f"{isl['w']}x{isl['h']} -> scala 2D {scale:.0%} -> {sw}x{sh}")
                    break
        if not placed_ok and verbose:
            print(f"    Island {PAGE_LABELS.get(isl['page'],str(isl['page']))} "
                  f"{isl['w']}x{isl['h']} non inserita (non c'e' spazio)")

    # ── 3. Costruisci texture da zero (sfondo neutro) ─────────────────────────
    canvas = np.full((CH, CW, 3), 20, dtype=np.uint8)  # sfondo scuro neutro

    for isl in placed:
        src_arr = np.array(load_image(page_textures[isl['page']]))
        u1,v1,u2,v2 = isl['bb']
        sh = isl['scaled_h']; h = isl['h']; w = isl['w']
        sw = isl.get('scaled_w', w)   # larghezza scalata (default = originale)
        src_chunk = src_arr[v1:v2+1, u1:u2+1]  # (h, w, 3)

        # Scala se necessario (verticale, orizzontale o entrambe)
        if h != sh or w != sw:
            chunk_img = Image.fromarray(src_chunk).resize((sw, sh), Image.LANCZOS)
            src_scaled = np.array(chunk_img)
        else:
            src_scaled = src_chunk

        # Costruisci chunk pulito: partenza da zero, copia SOLO pixel coperti dalla maschera
        # Poi riempi i gap con nearest-neighbor dei pixel coperti.
        # Cosi' NESSUN pixel di altri NPC sopravvive, neanche nelle righe vuote.
        # Maschera dell'isola, ridimensionata alle dimensioni scalate (sw x sh)
        isl_mask = isl['comp_mask'][v1:v2+1, u1:u2+1]  # (h, w) bool
        if h != sh or w != sw:
            mask_img = Image.fromarray(isl_mask.astype(np.uint8)*255).resize((sw, sh), Image.NEAREST)
            mask_scaled = np.array(mask_img) > 127
        else:
            mask_scaled = isl_mask

        clean = np.zeros_like(src_scaled)
        w_clean = clean.shape[1]
        # Prima passata: copia solo pixel coperti dalla maschera (scalata)
        for dy in range(sh):
            row_mask = mask_scaled[dy][:w_clean]
            cov = np.where(row_mask)[0]
            for cx in cov:
                clean[dy, int(cx)] = src_scaled[dy, int(cx)]
        # Seconda passata: nearest-neighbor per i pixel non coperti
        last_valid_row = None
        for dy in range(sh):
            row_mask = mask_scaled[dy][:w_clean]
            cov_cols = np.where(row_mask)[0]
            if len(cov_cols) > 0:
                not_cov = np.where(~row_mask)[0]
                for cx in not_cov:
                    cx = min(int(cx), w_clean-1)
                    nearest = int(cov_cols[np.argmin(np.abs(cov_cols - cx))])
                    clean[dy, cx] = clean[dy, min(nearest, w_clean-1)]
                last_valid_row = dy
            elif last_valid_row is not None:
                clean[dy] = clean[last_valid_row]  # copia ultima riga valida PULITA

        # Scrivi nel canvas
        du, dv = isl['dst_u'], isl['dst_v']
        w_actual = clean.shape[1]
        canvas[dv:dv+sh, du:du+w_actual] = clean

    # ── 4. UV remap: ogni polygon UV -> posizione nell'island patchata ─────────
    # Mappa: (page, u, v) -> island + offset
    page_islands = {}
    for isl in placed:
        page_islands.setdefault(isl['page'], []).append(isl)

    uv_remap = {}
    for p in target_polys:
        page = p['page']
        if page not in page_islands: continue
        for u, v in p['uvs']:
            if page not in uv_remap:
                uv_remap[page] = {}
            if (u,v) in uv_remap[page]: continue
            # Trova l'island che contiene questo UV
            best = None
            for isl in page_islands[page]:
                u1,v1,u2,v2 = isl['bb']
                if u1<=u<=u2 and v1<=v<=v2:
                    best = isl; break
            if best is None:
                # Fallback: island piu vicina per questa pagina
                best = min(page_islands[page],
                           key=lambda i: abs(u-(i['bb'][0]+i['bb'][2])//2) +
                                         abs(v-(i['bb'][1]+i['bb'][3])//2))
            sh = best['scaled_h']; h = best['h']
            sw = best.get('scaled_w', best['w']); w = best['w']
            # U: scala orizzontale se l'isola è stata scalata in larghezza
            if w != sw and w > 0:
                tu = (u - best['bb'][0]) / max(1, w-1)
                nu = best['dst_u'] + round(tu * (sw-1))
            else:
                nu = u + best['offset_u']
            nu = max(0, min(CW-1, nu))
            # V: scala verticale
            if h != sh and h > 0:
                t = (v - best['bb'][1]) / max(1, h-1)
                nv = best['dst_v'] + round(t * (sh-1))
            else:
                nv = v + best['offset_v']
            uv_remap[page][(u,v)] = (nu, max(0, min(CH-1, nv)))

    combined = Image.fromarray(canvas)
    return combined, uv_remap



    # Pagina primaria = TP15 (Gabe) se presente, altrimenti la piu usata
    page_poly_count={}
    for p in target_polys:
        pg=p['page']
        page_poly_count[pg]=page_poly_count.get(pg,0)+1
    primary=(960,0) if page_poly_count.get((960,0),0)>0 and (960,0) in page_textures else \
             max((pg for pg in page_poly_count if pg in page_textures and page_poly_count[pg]>0),
                 key=lambda pg:page_poly_count[pg], default=None)
    if primary is None or primary not in page_textures:
        return None, {}

    # ── Selezione smart del primary ──────────────────────────────────────────
    # Sceglie il primary che massimizza la qualita minima delle secondarie.
    # Regola: la faccia (page con nome 'face'/'npc_face') deve avere almeno
    # MIN_FACE_ROWS righe nel blocco libero, altrimenti si prova un primary diverso.
    MIN_FACE_ROWS = 32  # soglia minima accettabile per la faccia

    def compute_free_blocks(candidate_primary):
        """Calcola i blocchi liberi per un dato primary."""
        m=np.zeros((256,128),bool)
        rasterize_uv_triangles(target_polys,candidate_primary,m)
        used=set(int(y) for y in np.where(m.any(axis=1))[0])
        free=[y for y in range(256) if y not in used]
        grps=[]; s=None
        for y in sorted(free):
            if s is None: s=y; p=y
            elif y==p+1: p=y
            else: grps.append((s,p)); s=y; p=y
        if s is not None: grps.append((s,p))
        grps.sort(key=lambda g:g[1]-g[0],reverse=True)
        return grps

    pages_with_tex=[pg for pg in page_poly_count
                    if pg in page_textures and page_poly_count[pg]>0]

    # Identifica la pagina "faccia" (TP30 o equivalente)
    face_pages={(896,256),(832,256),(768,256)}
    face_pg=next((pg for pg in sec_pages_ordered
                  if pg in face_pages and pg in page_textures),None) \
            if 'sec_pages_ordered' in dir() else None
    # sec_pages_ordered non ancora definito — calcoliamolo ora per il check
    face_pg_candidate=next(
        (pg for pg in sorted(pages_with_tex,key=lambda p:page_poly_count[p],reverse=True)
         if pg in face_pages and pg!=primary),None)

    # Controlla se il primary attuale lascia abbastanza spazio per la faccia
    best_primary=primary
    if face_pg_candidate:
        groups_current=compute_free_blocks(primary)
        best_face_block=groups_current[0][1]-groups_current[0][0]+1 if groups_current else 0
        if best_face_block < MIN_FACE_ROWS:
            if verbose:
                print(f"  ⚠ Blocco libero per faccia: {best_face_block} righe "
                      f"(< {MIN_FACE_ROWS} minimo) — provo primary alternativo")
            # Prova ogni pagina come primary e scegli quella che da piu spazio alla faccia
            best_face_space=best_face_block
            for cand in pages_with_tex:
                if cand==primary or cand not in page_textures: continue
                grps=compute_free_blocks(cand)
                space=grps[0][1]-grps[0][0]+1 if grps else 0
                if space > best_face_space:
                    best_face_space=space; best_primary=cand
            if best_primary!=primary and verbose:
                print(f"  -> Primary cambiato: {PAGE_LABELS.get(primary,str(primary))} "
                      f"-> {PAGE_LABELS.get(best_primary,str(best_primary))} "
                      f"({best_face_space} righe libere per faccia)")
            elif best_face_space < MIN_FACE_ROWS:
                # Nessun primary migliore — usa proporzionale per tutti
                if verbose:
                    print(f"  -> Nessun primary soddisfa il minimo. Uso proporzionale.")
                # Fallback: proporzionale (stesso codice di hmd_pack_texture.py)
                total_poly=sum(page_poly_count[pg] for pg in pages_with_tex)
                MIN_R=10; extra=256-MIN_R*len(pages_with_tex)
                sorted_pg=sorted(pages_with_tex,key=lambda pg:page_poly_count[pg],reverse=True)
                cursor=0; alloc_fallback={}
                for i,pg in enumerate(sorted_pg):
                    m2=np.zeros((256,128),bool)
                    rasterize_uv_triangles(target_polys,pg,m2)
                    rows=[y for y in range(256) if m2[y].any()]
                    sv_min=min(rows) if rows else 0; sv_max=max(rows) if rows else 255
                    h=256-cursor if i==len(sorted_pg)-1 else \
                      max(MIN_R,round(page_poly_count[pg]/total_poly*extra)+MIN_R)
                    h=min(h,256-cursor-MIN_R*(len(sorted_pg)-i-1))
                    alloc_fallback[pg]=(cursor,cursor+h-1,sv_min,sv_max); cursor+=h
                # Costruisci combined con fallback proporzionale
                base_img2=Image.new('RGB',(128,256),(20,20,20)); combined_arr2=np.array(base_img2)
                uv_remap2={}
                for pg,(vs,ve,sv_min,sv_max) in alloc_fallback.items():
                    src2=np.array(load_image(page_textures[pg]))
                    sh=sv_max-sv_min+1; dh=ve-vs+1
                    chk=src2[sv_min:sv_max+1]
                    scaled=np.array(Image.fromarray(chk).resize((128,dh),Image.LANCZOS)) \
                           if sh!=dh else chk
                    combined_arr2[vs:ve+1]=scaled
                    uv_remap2[pg]={}
                    pvs=[v for p in target_polys if p['page']==pg for u,v in p['uvs']]
                    uv_min=min(pvs); uv_max=max(pvs)
                    for p in target_polys:
                        if p['page']!=pg: continue
                        for u,v in p['uvs']:
                            if (u,v) not in uv_remap2[pg]:
                                t=(v-uv_min)/max(1,uv_max-uv_min)
                                nv=vs+round(t*(ve-vs)); uv_remap2[pg][(u,v)]=(u,max(vs,min(ve,nv)))
                return Image.fromarray(combined_arr2), uv_remap2

    primary=best_primary

    # Base layer: primary texture
    base_img=load_image(page_textures[primary]).convert('RGB')
    combined=base_img.copy()
    combined_arr=np.array(combined)

    # Righe usate dal primary — pixel-perfect, UV invariate
    target_mask_primary=np.zeros((256,128),bool)
    rasterize_uv_triangles(target_polys,primary,target_mask_primary)
    primary_used_rows=set(int(y) for y in np.where(target_mask_primary.any(axis=1))[0])
    free_rows=[y for y in range(256) if y not in primary_used_rows]

    # Blocchi liberi ordinati per dimensione decrescente
    groups=[]; start=None
    for y in sorted(free_rows):
        if start is None: start=y; prev=y
        elif y==prev+1: prev=y
        else: groups.append((start,prev)); start=y; prev=y
    if start is not None: groups.append((start,prev))
    groups.sort(key=lambda g:g[1]-g[0], reverse=True)

    uv_remap={}  # {page: {(u,v): (new_u,new_v)}}
    free_idx=0   # indice nel blocco libero corrente (uno per pagina secondaria)

    # Pagine secondarie: un blocco solo ciascuna
    sec_pages_ordered=sorted(
        [pg for pg in page_poly_count if pg!=primary and pg in page_textures
         and page_poly_count.get(pg,0)>0],
        key=lambda pg:page_poly_count.get(pg,0), reverse=True)

    for sec_page in sec_pages_ordered:
        tex_path=page_textures[sec_page]

        # Rasterizza il target su questa pagina
        target_mask=np.zeros((256,128),bool)
        rasterize_uv_triangles(target_polys,sec_page,target_mask)

        # Rasterizza tutti gli ALTRI NPC sulla stessa pagina
        others_mask=np.zeros((256,128),bool)
        for npc_name,npc_polys in all_polys_by_npc.items():
            rasterize_uv_triangles(npc_polys,sec_page,others_mask)
        others_mask &= ~target_mask  # escludi il target stesso

        # Pixel esclusivi = solo target (per identificare la ZONA di bellhop)
        exclusive=target_mask & ~others_mask
        # Per l'EMBEDDING usiamo l'intera area del target (non solo esclusivi)
        # cosi non lasciamo zone bianche dove i pixel sono condivisi con altri NPC.
        # La "contaminazione" da altri NPC e' minima e preferibile alle zone bianche.
        embed_mask=binary_dilation(target_mask,iterations=6)  # margine generoso

        if not target_mask.any():
            if verbose: print(f"    {PAGE_LABELS.get(sec_page,str(sec_page))}: nessun pixel trovato")
            continue

        # Bounding box verticale dell'area TARGET (non solo esclusivi)
        target_rows=[y for y in range(256) if target_mask[y].any()]
        src_v_min,src_v_max=min(target_rows),max(target_rows)
        src_h=src_v_max-src_v_min+1

        # UV target V range (da tutti i polys del target su questa pagina)
        target_vs=[v for p in target_polys if p['page']==sec_page
                   for u,v in p['uvs']]
        uv_v_min,uv_v_max=min(target_vs),max(target_vs)

        # Scegli il blocco libero piu grande ancora disponibile
        if free_idx>=len(groups):
            if verbose: print(f"    {PAGE_LABELS.get(sec_page,str(sec_page))}: nessun blocco libero rimasto")
            continue
        dst_s,dst_e=groups[free_idx]; free_idx+=1
        dst_h=dst_e-dst_s+1

        if verbose:
            n_excl=exclusive.sum()
            ratio=100*dst_h//max(1,src_h)
            print(f"    {PAGE_LABELS.get(sec_page,str(sec_page))}: {n_excl} px esclusivi "
                  f"src V=[{src_v_min},{src_v_max}] ({src_h} righe) -> "
                  f"dst V=[{dst_s},{dst_e}] ({dst_h} righe, {ratio}%)")

        # Copia il contenuto esclusivo nel blocco destinazione
        src_img=load_image(tex_path)
        src_arr=np.array(src_img)
        src_chunk=src_arr[src_v_min:src_v_max+1,:,:]

        # Maschera target (solo i pixel di bellhop) scalata al chunk sorgente
        target_chunk=target_mask[src_v_min:src_v_max+1,:]  # bool 128xsrc_h
        # Dilata leggermente per coprire i bordi dei triangoli
        target_chunk_dilated=binary_dilation(target_chunk,iterations=4)

        # Crea chunk con SOLO i pixel del target — gli altri vengono
        # riempiti con il vicino piu prossimo per evitare colori spuri
        # nella palette (che sprecherebbero entries su altri NPC)
        clean_chunk=src_chunk.copy()
        # Per le colonne non coperte dal target, usa la media dei vicini coperti
        for row_idx in range(src_h):
            row_mask=target_chunk_dilated[row_idx]
            if not row_mask.any():
                # Riga completamente fuori dal target: usa riga adiacente
                if row_idx>0:
                    clean_chunk[row_idx]=clean_chunk[row_idx-1]
                continue
            # Colonne non coperte: fill dal vicino coperto piu vicino
            not_covered=~row_mask
            if not_covered.any():
                for u in np.where(not_covered)[0]:
                    # Trova colonna coperta piu vicina
                    covered_cols=np.where(row_mask)[0]
                    nearest=covered_cols[np.argmin(np.abs(covered_cols-u))]
                    clean_chunk[row_idx,u]=clean_chunk[row_idx,nearest]

        if src_h!=dst_h:
            chunk_img=Image.fromarray(clean_chunk).resize((128,dst_h),Image.LANCZOS)
            src_chunk_scaled=np.array(chunk_img)
        else:
            src_chunk_scaled=clean_chunk

        # Copia nel blocco destinazione (righe intere del chunk pulito)
        for dy in range(dst_h):
            src_row_idx=round(dy*src_h/max(1,dst_h-1))
            combined_arr[dst_s+dy, :]=src_chunk_scaled[dy, :]

        # Post-fill bordi
        for dy in range(dst_h):
            row=combined_arr[dst_s+dy]
            is_white=(row[:,0]>230)&(row[:,1]>230)&(row[:,2]>230)
            if is_white.any() and dy>0:
                combined_arr[dst_s+dy, is_white]=combined_arr[dst_s+dy-1, is_white]
        for dy in range(dst_h-1, 0, -1):
            row=combined_arr[dst_s+dy]
            is_white=(row[:,0]>230)&(row[:,1]>230)&(row[:,2]>230)
            if is_white.any():
                combined_arr[dst_s+dy, is_white]=combined_arr[dst_s+dy-1, is_white]

        # UV remap LINEARE (un solo blocco -> UV interpolation corretta sul PS1)
        uv_remap[sec_page]={}
        for p in target_polys:
            if p['page']!=sec_page: continue
            for u,v in p['uvs']:
                if (u,v) not in uv_remap[sec_page]:
                    # Mappa lineare: uv_v_min..uv_v_max -> dst_s..dst_e
                    if uv_v_max>uv_v_min:
                        t=(v-uv_v_min)/(uv_v_max-uv_v_min)
                        new_v=dst_s+round(t*(dst_e-dst_s))
                    else:
                        new_v=dst_s
                    new_v=max(dst_s,min(dst_e,new_v))
                    uv_remap[sec_page][(u,v)]=(u,new_v)

    combined=Image.fromarray(combined_arr)
    return combined, uv_remap

    combined=Image.fromarray(combined_arr)
    return combined, uv_remap

# ── Export OBJ con UV remappati ───────────────────────────────────────────────
def export_obj(target_polys, bones, div, uv_remap, out_path, tex_name, tex_w=128):
    T_adj=dict(T_EFF)
    chest_tx=next((b['tx'] for b in bones if b['name']=='Chest'),14)
    delta=chest_tx-14
    if delta:
        for bn in ['Head','LeftUpArm','LeftLowArm','LeftHand','RightUpArm','RightLowArm','RightHand']:
            if bn in T_adj: T_adj[bn]=(T_adj[bn][0],T_adj[bn][1]+delta,T_adj[bn][2])
    all_verts=[]; bone_start=[]; bnames=[]
    for b in bones:
        bone_start.append(len(all_verts))
        bnames.append(b['name'])
        T=T_adj.get(b['name'].strip('\x00\xcc\xff'),(0,0,0))
        for lx,ly,lz in b['verts']:
            all_verts.append(((lx+T[0])/64.,(-ly+T[1])/64.,(-lz+T[2])/64.))

    mtl_path=out_path.replace('.OBJ','.MTL')
    with open(mtl_path,'w',encoding='utf-8') as f:
        f.write(f"newmtl material\nKa 1 1 1\nKd 1 1 1\nmap_Kd {tex_name}\n")

    with open(out_path,'w',encoding='utf-8') as f:
        f.write(f"# level_npc_convert.py output\nmtllib {os.path.basename(mtl_path)}\n\n")
        for bi,name in enumerate(bnames):
            f.write(f"# {name}\n")
            bn=bone_start[bi+1] if bi+1<len(bone_start) else len(all_verts)
            for idx in range(bone_start[bi],bn):
                ox,oy,oz=all_verts[idx]
                f.write(f"v {ox:.6f} {oy:.6f} {oz:.6f}\n")
        f.write("\n")
        for p in target_polys:
            pg=p['page']
            for u,v in p['uvs']:
                if pg in uv_remap and (u,v) in uv_remap[pg]:
                    nu,nv=uv_remap[pg][(u,v)]
                else:
                    nu,nv=u,v
                f.write(f"vt {nu/tex_w:.6f} {1-nv/256.:.6f}\n")
        f.write("\ng body\nusemtl material\n")
        for pi,p in enumerate(target_polys):
            ub=pi*3+1; r=p['refs']
            f.write(f"f {r[0]+1}/{ub} {r[1]+1}/{ub+1} {r[2]+1}/{ub+2}\n")

# ── Patch HMD ────────────────────────────────────────────────────────────────
def patch_hmd_uvs(hmd_blob, target_polys, uv_remap, div, target_game='sf3', wide=False):
    hmd=bytearray(hmd_blob)
    pc=struct.unpack_from('<I',hmd,8)[0]//4
    src_stride = div
    dst_stride = 8 if target_game=='sf3' else 12
    need_recode = (src_stride != dst_stride)
    tsb = MPLAYER_TSB_WIDE if wide else MPLAYER_TSB

    for pi,p in enumerate(target_polys):
        o=0x24+pi*16
        if o+16>len(hmd): break
        pg=p['page']
        new_uvs=[]
        for u,v in p['uvs']:
            if pg in uv_remap and (u,v) in uv_remap[pg]:
                new_uvs.append(uv_remap[pg][(u,v)])
            else:
                new_uvs.append((u,v))
        (u0,v0),(u1,v1),(u2,v2)=new_uvs
        hmd[o]=u0; hmd[o+1]=v0; hmd[o+4]=u1; hmd[o+5]=v1; hmd[o+8]=u2; hmd[o+9]=v2
        struct.pack_into('<H',hmd,o+2,MPLAYER_CBA)
        struct.pack_into('<H',hmd,o+6,tsb)
        if need_recode:
            for voff in [o+10,o+12,o+14]:
                old=struct.unpack_from('<H',hmd,voff)[0]
                new_val=min(0xFFFF,(old//src_stride)*dst_stride)
                struct.pack_into('<H',hmd,voff,new_val)
    return hmd
    hmd=bytearray(hmd_blob)
    pc=struct.unpack_from('<I',hmd,8)[0]//4

    # Determina stride sorgente e destinazione
    src_stride = div                                      # 12=SF1/SF2, 8=SF3
    dst_stride = 8 if target_game=='sf3' else 12          # target richiesto

    # num/den per la scalatura vref:
    #   SF1/SF2->SF3: x8/12 = x2/3
    #   SF3->SF2:     x12/8 = x3/2
    #   stessa versione: nessuna scala
    need_recode = (src_stride != dst_stride)

    for pi,p in enumerate(target_polys):
        o=0x24+pi*16
        if o+16>len(hmd): break
        pg=p['page']
        new_uvs=[]
        for u,v in p['uvs']:
            if pg in uv_remap and (u,v) in uv_remap[pg]:
                new_uvs.append(uv_remap[pg][(u,v)])
            else:
                new_uvs.append((u,v))
        (u0,v0),(u1,v1),(u2,v2)=new_uvs
        hmd[o]=u0; hmd[o+1]=v0; hmd[o+4]=u1; hmd[o+5]=v1; hmd[o+8]=u2; hmd[o+9]=v2
        struct.pack_into('<H',hmd,o+2,MPLAYER_CBA)
        struct.pack_into('<H',hmd,o+6,MPLAYER_TSB)
        if need_recode:
            for voff in [o+10,o+12,o+14]:
                old=struct.unpack_from('<H',hmd,voff)[0]
                # Converte: indice_vertice = old // src_stride
                # Nuovo vref = indice_vertice * dst_stride
                new_val=min(0xFFFF,(old//src_stride)*dst_stride)
                struct.pack_into('<H',hmd,voff,new_val)
    return hmd

import importlib.util, pathlib

def _load_png_to_tim():
    """Carica png_to_tim.py dalla stessa directory dello script."""
    candidates=[
        pathlib.Path(__file__).parent/'png_to_tim.py',
        pathlib.Path('png_to_tim.py'),
    ]
    for c in candidates:
        if c.exists():
            spec=importlib.util.spec_from_file_location('png_to_tim',c)
            mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod
    return None

# ── PNG -> TIM ─────────────────────────────────────────────────────────────────
def png_to_tim(img, tim_path, wide=False):
    """Converti PIL Image -> TIM usando png_to_tim.py se disponibile."""
    mod=_load_png_to_tim()
    if mod:
        tmp_png=tim_path.replace('.TIM','.TMP.PNG')
        img.save(tmp_png,'PNG')
        mod.png_to_tim(tmp_png, tim_path, verbose=False, wide=wide)
        import os; os.remove(tmp_png)
    else:
        # Fallback interno (potrebbe non essere compatibile con tutti i tool)
        img_q=img.quantize(colors=256,method=Image.Quantize.MEDIANCUT,dither=0)
        palette=(img_q.getpalette()+[0]*768)[:768]
        pixels=bytes(img_q.tobytes())
        clut=bytearray(512)
        for i in range(256):
            r,g,b=palette[i*3],palette[i*3+1],palette[i*3+2]
            if r==0 and g==0 and b==0: r=1
            struct.pack_into('<H',clut,i*2,(r>>3)|((g>>3)<<5)|((b>>3)<<10))
        with open(tim_path,'wb') as f:
            f.write(struct.pack('<II',0x10,0x09))
            f.write(struct.pack('<I',8+4*2+512))
            f.write(struct.pack('<HHHH',768,480,256,1)); f.write(clut)
            f.write(struct.pack('<I',8+4*2+len(pixels)))
            f.write(struct.pack('<HHHH',512,0,64,256)); f.write(pixels)


# ── Estrazione texture da VLF.RFF + VRAM.HOG ─────────────────────────────────
# Palette default per TP standard (CLUT index in ZCLUT.BIN)
DEFAULT_TP_CLUT = {
    (960,   0): 30,   # TP15 — Gabe, penultima (di 32)
    (896,   0): 10,   # TP14 — corpo NPC
    (896, 256): 10,   # TP30 — faccia NPC
    (832, 256): 10,   # TP29 — multi-faccia SF3
    (768, 256): 10,   # TP28
    (384,   0): 10,   # speciale
}

# Mappa indirizzo pagina -> numero TP (per calcolare slot VLF)
# TP00-TP15: Y=0, X=0,64,...,960  |  TP16-TP31: Y=256, X=0,64,...,960
def page_to_tp_num(page_x, page_y):
    row = page_y // 256   # 0 o 1
    col = page_x // 64    # 0-15
    return row * 16 + col

def load_vram_hog(vram_path):
    """Carica VRAM.HOG e ritorna il dizionario dei file interni."""
    with open(vram_path,'rb') as f: d=f.read()
    nf=struct.unpack_from('<I',d,4)[0]; od=struct.unpack_from('<I',d,16)[0]
    oo=struct.unpack_from('<I',d,8)[0]; on=struct.unpack_from('<I',d,12)[0]
    fo=[struct.unpack_from('<I',d,oo+i*4)[0] for i in range(nf+1)]
    p,names=on,[]
    for _ in range(nf):
        while p<od and (d[p]<32 or d[p]>126): p+=1
        e=d.index(0,p); names.append(d[p:e].decode('ascii')); p=e+1
    return {names[i]:d[od+fo[i]:od+fo[i+1]] for i in range(nf)}

def extract_tp_from_vram(vram_files, page_xy, clut_idx, out_png=None):
    """
    Estrae una pagina texture direttamente da VRAM.HOG (TP##.BIN + ZCLUT.BIN).
    Nessun VLF.RFF necessario.

    vram_files : dict {nome: bytes} da load_vram_hog()
    page_xy    : (X,Y) indirizzo pagina (es. (896,256) per TP30)
    clut_idx   : indice CLUT in ZCLUT.BIN (0-31)
    out_png    : se fornito, salva PNG; altrimenti ritorna PIL Image

    TP number: TP##.BIN dove ## = (Y//256)*16 + (X//64)
    """
    tp_num = page_to_tp_num(*page_xy)
    tp_name = f"TP{tp_num:02d}.BIN"

    tp_data = vram_files.get(tp_name)
    if tp_data is None:
        print(f"  ✗ {tp_name} non trovato in VRAM.HOG (pagina {page_xy})")
        return None

    zclut = vram_files.get('ZCLUT.BIN', b'')
    n_cluts = len(zclut) // 512
    clut_idx = max(0, min(n_cluts - 1, clut_idx))

    pal = []
    for i in range(256):
        v = struct.unpack_from('<H', zclut, clut_idx * 512 + i * 2)[0]
        pal.append((0, 0, 0) if v == 0 else
                   ((v & 31) << 3, ((v >> 5) & 31) << 3, ((v >> 10) & 31) << 3))

    img = Image.new('RGB', (128, 256))
    pix = img.load()
    for y in range(256):
        for x in range(128):
            pix[x, y] = pal[tp_data[y * 128 + x]]

    if out_png:
        img.save(out_png, 'PNG')
    return img






    with open(vlf_path,'rb') as f: vlf=f.read()
    n_slots = len(vlf)//PAGE
    if n_slots == 0:
        print(f"  ✗ VLF vuoto o non valido: {vlf_path}"); return False

    vram_files = load_vram_hog(vram_path)
    zclut = vram_files.get('ZCLUT.BIN', b'')
    if not zclut:
        print(f"  ✗ ZCLUT.BIN non trovato in {vram_path}"); return False

    n_cluts = len(zclut)//512
    clut_idx = max(0, min(n_cluts-1, clut_idx))

    # Costruisci palette
    pal=[]
    for i in range(256):
        v=struct.unpack_from('<H',zclut,clut_idx*512+i*2)[0]
        pal.append((0,0,0) if v==0 else ((v&31)<<3,((v>>5)&31)<<3,((v>>10)&31)<<3))

    # Scegli slot: prova TP number direttamente, poi cerca il piu popolato
    slot_candidates = []
    if 0 <= tp_num < n_slots:
        slot_candidates.append(tp_num)
    # Aggiungi altri slot come fallback (escludi quelli gia provati)
    for s in range(n_slots):
        if s not in slot_candidates: slot_candidates.append(s)

    # Usa il primo slot valido (non tutto nero) tra i candidati
    best_slot = slot_candidates[0]
    best_nonzero = 0
    for slot in slot_candidates[:min(len(slot_candidates), 8)]:  # prova max 8 slot
        px = vlf[slot*PAGE:(slot+1)*PAGE]
        nonzero = sum(1 for b in px if b != 0)
        if slot == slot_candidates[0]:  # TP number diretto
            best_slot = slot; best_nonzero = nonzero; break
        if nonzero > best_nonzero:
            best_nonzero = nonzero; best_slot = slot

    px = vlf[best_slot*PAGE:(best_slot+1)*PAGE]
    img = Image.new('RGB',(128,256)); pix=img.load()
    for y in range(256):
        for x in range(128): pix[x,y]=pal[px[y*128+x]]
    img.save(out_png,'PNG')

    tp_label = PAGE_LABELS.get(page_xy, f"TP{tp_num}")
    print(f"  Estratto {tp_label} (slot VLF={best_slot}, CLUT={clut_idx}) -> {os.path.basename(out_png)}")
    return True

def list_vlf_slots(vlf_path, vram_path, clut_idx=10, out_dir='/tmp'):
    """Lista tutti gli slot del VLF con thumbnail 32x64."""
    PAGE=32768
    with open(vlf_path,'rb') as f: vlf=f.read()
    n_slots=len(vlf)//PAGE
    vram_files=load_vram_hog(vram_path)
    zclut=vram_files.get('ZCLUT.BIN',b'')
    pal=[]
    for i in range(256):
        v=struct.unpack_from('<H',zclut,clut_idx*512+i*2)[0] if zclut else 0
        pal.append((0,0,0) if v==0 else ((v&31)<<3,((v>>5)&31)<<3,((v>>10)&31)<<3))
    print(f"\nVLF: {n_slots} slot, TP=X/64-1 col + Y/256*16 row")
    for slot in range(n_slots):
        px=vlf[slot*PAGE:(slot+1)*PAGE]
        nonzero=sum(1 for b in px if b!=0)
        tp_num=slot  # assunzione diretta
        tp_x=(tp_num%16)*64; tp_y=(tp_num//16)*256
        tp_label=PAGE_LABELS.get((tp_x,tp_y),f"TP{tp_num}")
        print(f"  Slot {slot:2d}: TP{tp_num:2d} -> page({tp_x},{tp_y}) "
              f"{tp_label:12s}  px non-neri: {nonzero}")


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', help='Cartella con tutti i MODEL.HMD del livello')
    ap.add_argument('-o','--output', default=None, help='Directory output')
    ap.add_argument('--npc', default=None, help='Nome NPC target (selezione interattiva se omesso)')
    ap.add_argument('--make-vram', metavar='BLUEPRINT_VRAM.HOG', default=None,
                    help='Crea NOME_VRAM.HOG: TP15=texture baked, CLUT30=palette TIM. '
                         'Necessario per evitare corruzione al resume da pausa in gioco.')
    ap.add_argument('--all', action='store_true',
                    help='Batch mode: processa tutti gli NPC del livello')
    ap.add_argument('--slot', type=int, default=1, choices=[1,2],
                    help='Slot giocatore nell\'HOG: 1 (default) o 2 ([NOME]2SF*.HOG)')
    ap.add_argument('--make-hog', default=None, metavar='BLUEPRINT.HOG',
                    help='Blueprint HOG per la creazione del file finale. '
                         'Se omesso, usa SF3.HOG (target sf3) o SF1.HOG (target sf2) '
                         'dalla cartella degli script.')
    ap.add_argument('--wide', action='store_true',
                    help='Texture 256x256 (UV bake N colonne, TSB=136)')
    ap.add_argument('--wide-to-128', action='store_true',
                    help='Bake 256x256 poi ridimensiona a 128x256 (TSB=143, compatibile VLF)')
    ap.add_argument('--ultra', action='store_true',
                    help='[SPERIMENTALE] Texture 512x256: un TP per ogni 128px, '
                         'massima qualita\'. Richiede 2 TSB diversi.')
    ap.add_argument('--auto', action='store_true',
                    help='Auto: se tutto entra in 128x256 senza ridimensionamento usa standard, '
                         'altrimenti usa --wide-to-128 (default se nessuna modalita\' specificata)')
    ap.add_argument('--repack', action='store_true',
                    help='Island packing PURO: ogni pagina (TP15 inclusa) contribuisce solo le '
                         'isole UV usate, impacchettate per risoluzione massima. Scarta le parti '
                         'non usate (es. pixel Gabe non toccati dall\'NPC). Tutto in una pagina 128x256.')
    ap.add_argument('--no-modelx', action='store_true',
                    help='Omette MODELX.HMD dall\'HOG (stile Minigame, HOG piu\' piccolo)')

    # Texture: da VLF o file diretti
    tex=ap.add_argument_group('Texture')
    tex.add_argument('--vram', default=None, metavar='VRAM.HOG',
                     help='VRAM.HOG del livello (palette ZCLUT.BIN)')
    tex.add_argument('--vlf',  default=None, metavar='VLF.RFF',
                     help='VLF.RFF del livello (raw 128x256 8bpp per slot)')
    tex.add_argument('--list-vlf', action='store_true',
                     help='Elenca slot VLF con info TP, poi esce')
    tex.add_argument('--gabe', default=None, help='TIM/PNG TP15 (960,0) — Gabe/suit')
    tex.add_argument('--body', default=None, help='TIM/PNG TP14 (896,0) — corpo NPC')
    tex.add_argument('--face', default=None, help='TIM/PNG TP30 (896,256) — faccia NPC')
    tex.add_argument('--page', action='append', default=[], metavar='X,Y=path',
                     help='Pagina extra (es. 832,256=TP29.png)')
    tex.add_argument('--extract-tp', action='append', default=[], metavar='X,Y:CLUT[:SLOT]',
                     help='Estrai TP custom da VLF con CLUT specificata. '
                          'Formato: "X,Y:CLUT_IDX" oppure "X,Y:CLUT_IDX:VLF_SLOT". '
                          'Es: --extract-tp 960,0:30  --extract-tp 832,256:10:7  '
                          '--extract-tp 384,0:15:3 (maschera speciale)')

    ap.add_argument('--target', choices=['sf2','sf3'], default='sf3',
                    help='Gioco target MPLAYER (default: sf3)')

    # Modalita manuale palette
    man=ap.add_argument_group('Modalita manuale (pagine/palette non standard)')
    man.add_argument('--clut', action='append', default=[], metavar='TP_NUM=IDX',
                     help='Override CLUT per TP specifico. Es: --clut 15=30 --clut 14=10 '
                          '(default: TP15->CLUT30, tutti gli altri->CLUT10)')

    args=ap.parse_args()

    import re, tempfile

    # ── Risolvi CLUT overrides ─────────────────────────────────────────────────
    clut_overrides = {}   # {tp_num: clut_idx}
    for s in args.clut:
        try:
            tp, ci = s.split('=')
            clut_overrides[int(tp)] = int(ci)
        except ValueError:
            print(f"Formato --clut non valido: {s} (atteso TP_NUM=CLUT_IDX)"); sys.exit(1)

    def get_clut(page_xy):
        """Ritorna il CLUT index per una pagina, con eventuali override."""
        tp_num = page_to_tp_num(*page_xy)
        if tp_num in clut_overrides:
            return clut_overrides[tp_num]
        return DEFAULT_TP_CLUT.get(page_xy, 10)

    # ── Risolvi texture ────────────────────────────────────────────────────────
    page_textures = {}
    tmp_files     = []

    # Carica VRAM.HOG se fornito (priorita su VLF.RFF)
    vram_files = None
    if args.vram:
        print(f"\nCaricamento VRAM.HOG: {args.vram}")
        vram_files = load_vram_hog(args.vram)
        n_tp = sum(1 for k in vram_files if k.startswith('TP') and k.endswith('.BIN'))
        print(f"  {n_tp} pagine TP + ZCLUT.BIN trovati")

    # Auto-estrai TP standard da VRAM.HOG
    if vram_files:
        for page in sorted(DEFAULT_TP_CLUT.keys()):
            # Salta se gia fornita come file esplicito
            explicit = {(960,0):args.gabe,(896,0):args.body,(896,256):args.face}.get(page)
            if explicit:
                page_textures[page] = explicit
                continue
            # Controlla se esiste --page override
            if any(True for s in args.page
                   if re.match(r'(\d+),(\d+)=',s) and
                   (int(re.match(r'(\d+),(\d+)=',s).group(1)),
                    int(re.match(r'(\d+),(\d+)=',s).group(2)))==page):
                continue
            clut_idx = get_clut(page)
            tmp = tempfile.mktemp(suffix='.PNG')
            img = extract_tp_from_vram(vram_files, page, clut_idx)
            if img:
                img.save(tmp, 'PNG')
                page_textures[page] = tmp
                tmp_files.append(tmp)
                tp_num = page_to_tp_num(*page)
                print(f"  TP{tp_num:02d} page{page} -> CLUT {clut_idx} -> {tmp.split('/')[-1]}")



    # --extract-tp: TP custom con CLUT specifica (maschere, abiti speciali, ecc.)
    for spec in args.extract_tp:
        m=re.match(r'(\d+),(\d+):(\d+)(?::(\d+))?',spec)
        if not m:
            print(f"  ✗ Formato non valido: {spec}  (atteso X,Y:CLUT o X,Y:CLUT:SLOT)"); continue
        page=(int(m.group(1)),int(m.group(2)))
        clut_idx=int(m.group(3))
        slot_explicit=int(m.group(4)) if m.group(4) else None

        if not args.vram or not args.vlf:
            print("  ✗ --extract-tp richiede --vram e --vlf"); continue

        if slot_explicit is not None:
            # Slot VLF esplicito
            PAGE_SZ=32768
            with open(args.vlf,'rb') as f: vlf=f.read()
            vf=load_vram_hog(args.vram); zclut=vf.get('ZCLUT.BIN',b'')
            n_cluts=len(zclut)//512; ci=max(0,min(n_cluts-1,clut_idx))
            pal=[]
            for i in range(256):
                v=struct.unpack_from('<H',zclut,ci*512+i*2)[0] if zclut else 0
                pal.append((0,0,0) if v==0 else ((v&31)<<3,((v>>5)&31)<<3,((v>>10)&31)<<3))
            spx=vlf[slot_explicit*PAGE_SZ:(slot_explicit+1)*PAGE_SZ]
            img=Image.new('RGB',(128,256)); pix_=img.load()
            for y in range(256):
                for x in range(128): pix_[x,y]=pal[spx[y*128+x]]
            tmp=tempfile.mktemp(suffix='.PNG'); img.save(tmp,'PNG')
            label=PAGE_LABELS.get(page,f"({page[0]},{page[1]})")
            print(f"  Estratto {label} (slot={slot_explicit}, CLUT={clut_idx}) -> {os.path.basename(tmp)}")
            page_textures[page]=tmp; tmp_files.append(tmp)
        else:
            tmp=tempfile.mktemp(suffix='.PNG')
            if extract_tp_from_vlf(args.vlf,args.vram,page,clut_idx,tmp):
                page_textures[page]=tmp; tmp_files.append(tmp)

    # File texture espliciti (priorita su estrazione auto)
    if args.gabe: page_textures[(960,  0)]=args.gabe
    if args.body: page_textures[(896,  0)]=args.body
    if args.face: page_textures[(896,256)]=args.face
    for s in args.page:
        m=re.match(r'(\d+),(\d+)=(.+)',s)
        if m: page_textures[(int(m.group(1)),int(m.group(2)))]=m.group(3)

    # ── Auto-rilevamento TP??.PNG nella cartella del livello ─────────────────
    # Cerca TP00.PNG ... TP31.PNG e mappa automaticamente al page address
    def tp_page(tp_num):
        """Converte numero TP in page address (x,y)."""
        return ((tp_num & 0xF) * 64, ((tp_num >> 4) & 1) * 256)

    tp_auto_found = []
    for fname in os.listdir(args.folder):
        m = re.match(r'[Tt][Pp](\d{2})\.(png|PNG)$', fname)
        if not m: continue
        tp_num = int(m.group(1))
        page = tp_page(tp_num)
        fpath = os.path.join(args.folder, fname)
        if page not in page_textures:  # non sovrascrive espliciti
            page_textures[page] = fpath
            tp_auto_found.append(f"TP{tp_num:02d}→{page}")

    if tp_auto_found:
        print(f"  Auto-texture: {', '.join(tp_auto_found)}")

    # 1. Scansiona cartella
    print(f"\nScansione: {args.folder}")
    found=scan_level_folder(args.folder)
    if not found:
        print("Nessun MODEL.HMD trovato!"); sys.exit(1)

    print(f"\nNPC trovati ({len(found)}):")
    for i,(name,path) in enumerate(found):
        print(f"  {i+1:2d}. {name}")

    # 2. Seleziona target(s)
    # Se --npc specificato → singolo NPC
    # Se --all o nessun --npc → batch: tutti gli NPC del livello
    batch_mode = getattr(args,'all',False) or (not args.npc)

    if batch_mode and len(found) > 1:
        # Definisci out_dir prima del loop batch
        out_dir = args.output if args.output else os.path.join(args.folder, 'MPLAYER_OUT')
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nModalità batch: processing tutti i {len(found)} NPC")
        print(f"Output: [NOME]2{args.target.upper()}.HOG")

        # Richiama se stesso per ogni NPC con --npc esplicito
        import subprocess
        success, failed = [], []
        for name,path in found:
            print(f"\n{'='*50}\n>>> {name}")
            cmd = [sys.executable, __file__,
                   args.folder,
                   '--npc', name,
                   '--target', args.target,
                   '--slot', str(getattr(args,'slot',2)),
                   '-o', out_dir]
            # Passa tutti gli argomenti texture
            if args.vram:   cmd += ['--vram', args.vram]
            if args.face:   cmd += ['--face', args.face]
            if args.body:   cmd += ['--body', args.body]
            if args.gabe:   cmd += ['--gabe', args.gabe]
            if getattr(args,'no_modelx',False): cmd += ['--no-modelx']
            if getattr(args,'make_vram',None):  cmd += ['--make-vram', args.make_vram]
            if getattr(args, 'wide_to_128', False): cmd += ['--wide-to-128']
            if getattr(args, 'wide', False):        cmd += ['--wide']
            if getattr(args, 'auto', False):        cmd += ['--auto']
            if getattr(args, 'repack', False):      cmd += ['--repack']
            if getattr(args, 'ultra', False):       cmd += ['--ultra']
            if getattr(args, 'make_hog', None):     cmd += ['--make-hog', args.make_hog]
            for pg_arg in getattr(args,'page',[]):
                cmd += ['--page', f"{pg_arg[0][0]},{pg_arg[0][1]}={pg_arg[1]}"]
            # I TP??.PNG nella cartella verranno auto-rilevati dal subprocess

            r = subprocess.run(cmd, capture_output=True, text=True)
            # Stampa output del subprocess (viene catturato e ristampato)
            if r.stdout: print(r.stdout, end='')
            if r.stderr: print(r.stderr, end='', file=sys.stderr)
            (success if r.returncode==0 else failed).append(name)

        print(f"\n{'='*50}")
        print(f"Batch completato: {len(success)}/{len(found)} OK")
        if failed: print(f"  Falliti: {', '.join(failed)}")
        sys.exit(0 if not failed else 1)

    if args.npc:
        target_idx=next((i for i,(n,p) in enumerate(found)
                         if args.npc.lower() in n.lower()), None)
        if target_idx is None:
            print(f"NPC '{args.npc}' non trovato!"); sys.exit(1)
    elif len(found)==1:
        target_idx=0
    else:
        try:
            choice=int(input("\nScegli NPC da convertire (numero): "))-1
            if not 0<=choice<len(found): raise ValueError
            target_idx=choice
        except (ValueError,EOFError):
            print("Selezione non valida"); sys.exit(1)

    target_name, target_path=found[target_idx]
    print(f"\nTarget: {target_name}")

    # 3. Carica tutti i modelli
    print("\nCaricamento modelli...")
    all_polys_by_npc={}
    target_polys=target_bones=target_div=None

    for name,path in found:
        with open(path,'rb') as f: blob=f.read()
        result=parse_hmd(blob)
        if result[0] is None: continue
        polys,bones,div=result
        all_polys_by_npc[name]=polys
        if name==target_name:
            target_polys,target_bones,target_div=polys,bones,div
            target_blob=blob
        print(f"  {name}: {len(polys)} poly, div={div}")

    if target_polys is None:
        print("Errore nel parsing del target!"); sys.exit(1)

    # 4. Statistiche pagine target
    from collections import Counter
    page_counts=Counter(p['page'] for p in target_polys)
    print(f"\nPagine texture usate da {target_name}:")
    for pg,cnt in page_counts.most_common():
        label=PAGE_LABELS.get(pg,str(pg))
        has_tex='✓' if pg in page_textures else '✗ (nessuna texture)'
        print(f"  {label:<12} {cnt:3d} poly  {has_tex}")

    # 5. Costruisci combined texture con pixel esclusivi
    print("\nAnalisi pixel esclusivi per pagina...")
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:
        print("  scipy non disponibile, installa con: pip install scipy")
        sys.exit(1)

    # ── Modalita' single-texture ──────────────────────────────────────────────
    # Se l'utente fornisce solo --gabe (TP15 modificato manualmente con i pezzi
    # dell'NPC gia' dipinti), usalo direttamente per tutti i poligoni.
    # TSB di tutti i poly -> TP15, UV invariate, nessun remapping.
    # L'utente ha gia' posizionato faccia/dettagli nelle coordinate corrette.
    only_gabe = (args.gabe and not args.face and not args.body and
                 not any(True for s in args.page))
    if only_gabe:
        print("  Modalita' single-texture: --gabe usato per tutti i poligoni")
        print("  UV invariate — l'utente ha gia' dipinto i pezzi nel TP15")
        combined = load_image(args.gabe).convert('RGB')
        uv_remap = {}   # nessun remap: le UV puntano gia' alle posizioni corrette
    else:
        # ── Repack: island packing PURO, salta l'auto-detection ───────────────
        # Forza 128x256 pagina singola, ogni pagina contribuisce solo le isole usate
        if getattr(args,'repack',False):
            print(f"  Repack: island packing puro a piena risoluzione (128x256)")
            args.wide = False
            args.wide_to_128 = False
        # ── Auto-mode: decide la strategia ottimale basandosi sui pixel reali ──
        elif getattr(args,'auto',False) or (
                not args.wide and not getattr(args,'wide_to_128',False)
                and not getattr(args,'ultra',False)):

            # Conta pixel NPC totali su tutte le pagine (UV bake rapido)
            total_npc_px = 0
            n_pages = len([pg for pg in set(p['page'] for p in target_polys)
                           if pg in page_textures])
            for pg in set(p['page'] for p in target_polys):
                if pg not in page_textures: continue
                bk = np.zeros((256, 128), bool)
                for p in target_polys:
                    if p['page'] != pg: continue
                    (u0,v0),(u1,v1),(u2,v2) = p['uvs']
                    for sv in range(max(0,min(v0,v1,v2)),min(255,max(v0,v1,v2))+1):
                        hits=[]
                        for (ua,va),(ub,vb) in [((u0,v0),(u1,v1)),((u1,v1),(u2,v2)),((u2,v2),(u0,v0))]:
                            if (va<=sv<vb) or (vb<=sv<va):
                                hits.append(ua+(ub-ua)*(sv-va)/(vb-va))
                        if len(hits)>=2:
                            for su in range(max(0,int(min(hits))),min(127,int(max(hits)))+1):
                                bk[sv,su]=True
                total_npc_px += bk.sum()

            # Se i pixel NPC entrano in 128x256 → usa island packing standard (massima qualità)
            # Se non entrano → usa wide-to-128 (UV bake a piena risoluzione poi ridimensiona)
            # ECCEZIONI → sempre wide-to-128:
            #   1. Personaggi con TP15 (Gabe suit) → layout 3 colonne
            #   2. Pagine con UV > 127 (es. TP28 U=[112,143]) → non gestibili in 128px
            #   3. Tre o più pagine texture → island packing non ha spazio sufficiente
            CANVAS_128 = 32768  # 128x256
            GABE_PAGE  = (960, 0)
            has_gabe   = any(p['page'] == GABE_PAGE for p in target_polys)
            has_wide_uv = any(u > 127 for p in target_polys for u,v in p['uvs']
                              if p['page'] in page_textures)

            if has_gabe and n_pages > 1:
                # Gabe suit + pagine secondarie → 3 colonne (body, face)
                print(f"  Auto: TP15 + {n_pages-1} pag. secondarie → wide-to-128")
                args.wide_to_128 = True
            elif has_gabe:
                # Solo TP15, nessuna pagina secondaria → standard
                print(f"  Auto: solo TP15 (Gabe) → 128x256 standard")
            elif has_wide_uv and n_pages > 1:
                # UV > 127 con più pagine → colonne wide necessarie
                print(f"  Auto: UV > 127 + {n_pages} pagine → wide-to-128")
                args.wide_to_128 = True
            elif total_npc_px <= CANVAS_128:
                # Tutto entra in 128x256
                print(f"  Auto: {total_npc_px}px in {n_pages} pagine → 128x256 standard")
            else:
                # Non entra in 128x256
                print(f"  Auto: {total_npc_px}px in {n_pages} pagine → wide-to-128")
                args.wide_to_128 = True

        # ── Ultra mode (512x256): CW=512 ──────────────────────────────────────
        if getattr(args,'ultra',False):
            args.wide = True  # usa il meccanismo wide ma con CW=512

        # --wide-to-128: attiva wide per il bake 256x256
        if getattr(args,'wide_to_128',False):
            args.wide = True

        CW_ULTRA = 512 if getattr(args,'ultra',False) else (256 if args.wide else 128)

        combined, uv_remap=build_exclusive_texture(
            target_polys, all_polys_by_npc, page_textures,
            verbose=True, wide=args.wide,
            CW_override=CW_ULTRA if getattr(args,'ultra',False) else None,
            repack=getattr(args,'repack',False))

    if combined is None:
        print("Errore: nessuna texture primaria disponibile!"); sys.exit(1)

    # Inpaint zone bianche
    arr=np.array(combined)
    mask=(arr[:,:,0]>230)&(arr[:,:,1]>230)&(arr[:,:,2]>230)
    print(f"\nInpaint: {mask.sum()} pixel bianchi...")
    filled=arr.copy()
    for _ in range(40):
        if not mask.any(): break
        ys,xs=np.where(mask); new=filled.copy()
        for y,x in zip(ys,xs):
            nb=[filled[y+dy,x+dx]
                for dy,dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
                if 0<=y+dy<256 and 0<=x+dx<128 and not mask[y+dy,x+dx]]
            if nb: new[y,x]=np.mean(nb,axis=0).astype(np.uint8); mask[y,x]=False
        filled=new
    combined=Image.fromarray(filled)
    print(f"  Rimasti: {mask.sum()}")

    # 6. Output
    out_dir=args.output or '.'
    os.makedirs(out_dir,exist_ok=True)
    safe_name=os.path.basename(target_name).replace('/','_').replace('\\','_')

    # --wide-to-128: texture già buildata come 256x256, ora dimezza
    tex_png=os.path.join(out_dir,f"{safe_name}_TEXTURE.PNG")
    tex_tim=os.path.join(out_dir,f"{safe_name}_AMODEL.TIM")
    out_obj=os.path.join(out_dir,f"{safe_name}.OBJ")
    out_hmd=os.path.join(out_dir,f"{safe_name}_MODEL.HMD")

    if getattr(args,'wide_to_128',False):
        # Salva 256x256 come riferimento, poi ridimensiona a 128x256
        combined.save(tex_png.replace('_TEXTURE.PNG','_TEXTURE_256.PNG'))
        combined_128 = combined.resize((128, 256), Image.LANCZOS)
        combined_128.save(tex_png)
        print(f"\nTexture: {tex_png} (128x256 da bake 256x256)")
        png_to_tim(combined_128, tex_tim, wide=False)

        # Costruisci uv_remap_128: dimezza new_u per TUTTE le pagine
        # (secondarie: new_u era in [0,255] → /2 = [0,127])
        uv_remap_128 = {}
        for pg, mapping in uv_remap.items():
            uv_remap_128[pg] = {(u,v): (new_u//2, new_v)
                                 for (u,v),(new_u,new_v) in mapping.items()}
        # Pagine primarie (suit, non in uv_remap): U era in [0,127] → /2 = [0,63]
        for p in target_polys:
            pg = p['page']
            if pg not in uv_remap_128:
                uv_remap_128[pg] = {}
            for u,v in p['uvs']:
                if (u,v) not in uv_remap_128[pg]:
                    uv_remap_128[pg][(u,v)] = (u//2, v)

        export_obj(target_polys, target_bones, target_div, uv_remap_128,
                   out_obj, os.path.basename(tex_png), tex_w=128)
        args.wide = False   # HMD usa TSB=143
        uv_remap = uv_remap_128  # usa remap dimezzato anche per l'HMD
    else:
        combined.save(tex_png)
        print(f"\nTexture: {tex_png}")
        png_to_tim(combined, tex_tim, wide=args.wide)
        export_obj(target_polys, target_bones, target_div, uv_remap,
                   out_obj, os.path.basename(tex_png),
                   tex_w=256 if args.wide else 128)
    print(f"TIM:     {tex_tim}")
    print(f"OBJ:     {out_obj}")

    patched_hmd=patch_hmd_uvs(target_blob, target_polys, uv_remap,
                               target_div, args.target, wide=args.wide)
    with open(out_hmd,'wb') as f: f.write(patched_hmd)
    print(f"HMD:     {out_hmd}  [target: {args.target.upper()}]")

    # Pulizia file temporanei (estratti da VLF)
    for tmp in tmp_files:
        try: os.remove(tmp)
        except: pass

    # ── Creazione HOG: sempre entrambe le versioni SF2 e SF3 ──────────────────
    # Cerca blueprint nella cartella degli script: SF1.HOG (per SF2) e SF3.HOG
    script_dir = pathlib.Path(__file__).parent
    blueprints = {
        'SF2': args.make_hog if (args.make_hog and 'SF1' in args.make_hog.upper()) else str(script_dir/'SF1.HOG'),
        'SF3': args.make_hog if (args.make_hog and 'SF3' in args.make_hog.upper()) else str(script_dir/'SF3.HOG'),
    }

    # Carica hog_io una volta sola
    hog_mod = None
    for c in [script_dir/'hog_io.py', pathlib.Path('hog_io.py')]:
        if c.exists():
            spec = importlib.util.spec_from_file_location('hog_io', c)
            hog_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hog_mod)
            break
    if hog_mod is None:
        print("  ⚠ hog_io.py non trovato — HOG non creato")
    else:
        for game, bp in blueprints.items():
            if not pathlib.Path(bp).exists():
                continue
            try:
                # Per SF2 serve HMD con vref x12, per SF3 con vref x8
                # Se il target corrente corrisponde usiamo l'HMD già generato,
                # altrimenti ne creiamo uno convertito al volo
                if game == args.target.upper():
                    hmd_for_game = out_hmd
                else:
                    alt_target = 'sf2' if game == 'SF2' else 'sf3'
                    alt_hmd_path = out_hmd.replace(
                        f'_MODEL.HMD', f'_MODEL_{game}.HMD')
                    alt_blob = patch_hmd_uvs(target_blob, target_polys, uv_remap,
                                             target_div, alt_target)
                    with open(alt_hmd_path, 'wb') as f: f.write(alt_blob)
                    hmd_for_game = alt_hmd_path

                out_hog = os.path.join(out_dir,
                                       f"{safe_name.upper()}{getattr(args,'slot',1)}{game}.HOG")
                files_hog = {'MODEL.HMD': hmd_for_game, 'AMODEL.TIM': tex_tim}
                if not getattr(args,'no_modelx',False):
                    files_hog['MODELX.HMD'] = hmd_for_game
                hog_mod.pack_hog_from_blueprint(bp, files_hog, out_hog)
                print(f"HOG{game}:  {out_hog} ({'no MODELX' if getattr(args,'no_modelx',False) else '3 file'})")
            except Exception as e:
                print(f"  ⚠ Errore HOG {game}: {e}")

    # ── Crea VRAM.HOG per evitare corruzione al resume da pausa ──────────────
    if args.make_vram and hog_mod:
        try:
            vram_blueprint = args.make_vram
            with open(tex_tim,'rb') as f: tim_d=f.read()
            clut_block_len = struct.unpack_from('<I',tim_d,8)[0]
            clut_bytes = tim_d[8+12:8+12+512]
            pix_off = 8+clut_block_len
            pix_bytes = tim_d[pix_off+12:pix_off+12+32768]

            vram_hog = hog_mod.read_hog(vram_blueprint)
            zclut = bytearray(vram_hog['ZCLUT.BIN'])
            zclut[30*512:31*512] = clut_bytes

            import tempfile
            tmp2 = tempfile.mkdtemp()
            with open(f'{tmp2}/TP15.BIN','wb') as f: f.write(pix_bytes)
            with open(f'{tmp2}/ZCLUT.BIN','wb') as f: f.write(bytes(zclut))

            out_vram_hog = os.path.join(out_dir, f"{safe_name.upper()}_VRAM.HOG")
            hog_mod.pack_hog_from_blueprint(
                vram_blueprint,
                {'TP15.BIN': f'{tmp2}/TP15.BIN',
                 'ZCLUT.BIN': f'{tmp2}/ZCLUT.BIN'},
                out_vram_hog)
            print(f"VRAM.HOG: {out_vram_hog}  (TP15+CLUT30 aggiornati)")
        except Exception as e:
            print(f"  Avviso --make-vram: {e}")

    print(f"\n✓ Completato! File in: {out_dir}\n")