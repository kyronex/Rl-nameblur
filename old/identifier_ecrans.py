# identifier_ecrans.py

import dxcam
import screeninfo

# ══════════════════════════════════════
# MÉTHODE 1 : screeninfo (détails)
# ══════════════════════════════════════
print("=" * 50)
print("  VOS ÉCRANS DÉTECTÉS")
print("=" * 50)

for i, m in enumerate(screeninfo.get_monitors()):
    print(f"""
  Écran {i}
  ───────
  Nom        : {m.name}
  Résolution : {m.width} x {m.height}
  Position   : x={m.x}, y={m.y}
  Principal  : {"✅ OUI" if m.is_primary else "❌ NON"}
""")

# ══════════════════════════════════════
# MÉTHODE 2 : dxcam (ce qu'il voit)
# ══════════════════════════════════════
print("=" * 50)
print("  DXCAM - ÉCRANS DISPONIBLES")
print("=" * 50)

for i in range(3):  # teste jusqu'à 3 écrans
    try:
        cam = dxcam.create(output_idx=i)
        frame = cam.grab()
        if frame is not None:
            print(f"""
  output_idx={i}
  ─────────────
  Taille capturée : {frame.shape[1]} x {frame.shape[0]}
  ✅ Disponible
""")
        cam.release()
    except:
        print(f"\n  output_idx={i} → ❌ Pas d'écran\n")
